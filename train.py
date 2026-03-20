"""
OFT Fine-tuning of Qwen2.5-1.5B on GSM8K

Orthogonal Fine-Tuning (OFT) preserves hyperspherical energy of pretrained weights
by learning orthogonal transformation matrices. Unlike LoRA (additive low-rank updates),
OFT applies multiplicative orthogonal updates: W' = R * W, where R is orthogonal (R^T R = I).

Reference: Qiu et al., "Controlling Text-to-Image Diffusion by Orthogonal Fine-tuning" (NeurIPS 2023)
PEFT docs: https://huggingface.co/docs/peft/main/en/package_reference/oft
"""

import json
import os
import re
import inspect

import torch
from datasets import load_dataset
from peft import OFTConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B"   # ~1.5B params — closest public Qwen model to "1.3B"
OUTPUT_DIR = "./output/oft-qwen-gsm8k"
LOG_FILE = "./output/training_logs.json"

OFT_RANK = 8          # block size (r) for the orthogonal matrix decomposition
MAX_SEQ_LEN = 512
BATCH_SIZE = 4
GRAD_ACC = 4          # effective batch = 16
LR = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
SEED = 42


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Solve the following problem step by step. "
    "At the end, write your final answer after '####'."
)


def format_example(example):
    """Format a GSM8K example into a prompt+answer string for SFT."""
    question = example["question"].strip()
    answer = example["answer"].strip()   # contains chain-of-thought + "#### <number>"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


# ---------------------------------------------------------------------------
# Callback: save loss to JSON for later plotting
# ---------------------------------------------------------------------------
class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.records.append(
                {
                    "step": state.global_step,
                    "epoch": round(state.epoch, 4) if state.epoch else None,
                    "loss": logs["loss"],
                    "learning_rate": logs.get("learning_rate"),
                }
            )
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w") as f:
                json.dump(self.records, f, indent=2)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        if "eval_loss" in metrics:
            self.records.append(
                {
                    "step": state.global_step,
                    "epoch": round(state.epoch, 4) if state.epoch else None,
                    "eval_loss": metrics["eval_loss"],
                }
            )
            with open(self.log_file, "w") as f:
                json.dump(self.records, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Tokenizer ---
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Base model ---
    print(f"Loading base model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # --- OFT config ---
    # target_modules: apply OFT to the attention projection layers.
    # OFT wraps each target linear layer with an orthogonal matrix R:
    #   output = R @ (W @ input)
    # r controls the block-diagonal structure: larger r = more expressive but more params.
    oft_config = OFTConfig(
        r=OFT_RANK,
        oft_block_size=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        module_dropout=0.0,
        init_weights=True,          # initialise R = Identity (no change at start)
        coft=False,                 # set True for Constrained OFT (Cayley parameterisation)
        eps=6e-5,                   # epsilon for constrained OFT (only used if coft=True)
        block_share=False,          # share one block across all layers (more param-efficient)
    )

    model = get_peft_model(model, oft_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    print("Loading GSM8K dataset...")
    raw = load_dataset("gsm8k", "main")
    train_ds = raw["train"].map(lambda ex: {"text": format_example(ex)})
    eval_ds = raw["test"].map(lambda ex: {"text": format_example(ex)})

    # Use a small eval subset to keep training fast
    eval_ds = eval_ds.select(range(200))

    # --- Training args ---
    # Keep compatibility across TRL versions where some SFTConfig fields moved
    # between SFTConfig and SFTTrainer.
    sft_config_kwargs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACC,
        "learning_rate": LR,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "bf16": True,
        "logging_steps": 10,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",       # set to "wandb" if you want W&B logging
        "seed": SEED,
        "max_seq_length": MAX_SEQ_LEN,
        "dataset_text_field": "text",
        "packing": False,
    }
    sft_config_signature = inspect.signature(SFTConfig.__init__).parameters
    filtered_sft_kwargs = {
        k: v for k, v in sft_config_kwargs.items() if k in sft_config_signature
    }
    training_args = SFTConfig(**filtered_sft_kwargs)

    # --- Trainer ---
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "callbacks": [LossLoggerCallback(LOG_FILE)],
    }
    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
    if "max_seq_length" in trainer_signature:
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LEN
    if "dataset_text_field" in trainer_signature:
        trainer_kwargs["dataset_text_field"] = "text"
    if "packing" in trainer_signature:
        trainer_kwargs["packing"] = False
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    print("Starting OFT fine-tuning...")
    trainer.train()

    # --- Save final adapter ---
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved OFT adapter to {final_path}")


if __name__ == "__main__":
    main()
