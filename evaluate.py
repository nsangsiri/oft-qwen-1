"""
Evaluate fine-tuned (OFT) and baseline Qwen2.5-1.5B on GSM8K test set.

Computes exact-match accuracy by extracting the final number after "####"
from both the model's output and the ground-truth answer.

Usage:
    # Evaluate fine-tuned model
    python evaluate.py --model_path ./output/oft-qwen-gsm8k/final --tag oft

    # Evaluate baseline (no fine-tuning)
    python evaluate.py --model_path Qwen/Qwen2.5-1.5B --tag baseline --no_adapter

    # Run both and compare
    python evaluate.py --compare
"""

import argparse
import json
import os
import re
import sys

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B"
RESULTS_DIR = "./output/results"

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Solve the following problem step by step. "
    "At the end, write your final answer after '####'."
)


def extract_answer(text: str) -> str | None:
    """Extract the numeric answer after '####' from model output or ground truth."""
    # Ground truth format: "... #### 42" or "... #### 1,234"
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        # Normalise: remove commas, strip whitespace
        return match.group(1).replace(",", "").strip()
    return None


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_model(model_path: str, use_adapter: bool, base_model_id: str):
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_adapter:
        print(f"Loading OFT adapter from {model_path}...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()   # merge OFT weights for faster inference

    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate(model, tokenizer, dataset, max_new_tokens: int = 256, num_samples: int = None):
    device = next(model.parameters()).device
    correct = 0
    total = 0
    results = []

    samples = dataset if num_samples is None else dataset.select(range(num_samples))

    for example in tqdm(samples, desc="Evaluating"):
        question = example["question"].strip()
        gt_answer_text = example["answer"].strip()
        gt_answer = extract_answer(gt_answer_text)

        prompt = build_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred_answer = extract_answer(generated_text)

        is_correct = pred_answer is not None and pred_answer == gt_answer
        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "question": question,
                "ground_truth": gt_answer,
                "prediction": pred_answer,
                "generated_text": generated_text,
                "correct": is_correct,
                "level": example.get("level", "unknown"),
            }
        )

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def run_eval(model_path: str, use_adapter: bool, tag: str, num_samples: int = None):
    model, tokenizer = load_model(model_path, use_adapter, MODEL_ID)

    print("Loading GSM8K test set...")
    test_ds = load_dataset("gsm8k", "main", split="test")

    accuracy, results = evaluate(model, tokenizer, test_ds, num_samples=num_samples)

    print(f"\n{'='*50}")
    print(f"  [{tag}]  GSM8K Accuracy: {accuracy*100:.2f}%  ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"{'='*50}\n")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{tag}_results.json")
    with open(out_path, "w") as f:
        json.dump({"accuracy": accuracy, "num_correct": sum(r["correct"] for r in results),
                   "total": len(results), "results": results}, f, indent=2)
    print(f"Results saved to {out_path}")

    return accuracy, results


def print_qualitative(baseline_results, oft_results, n: int = 5, show_reasoning: bool = False):
    """Print side-by-side qualitative examples where OFT is correct but baseline is not."""
    print("\n" + "="*70)
    print("QUALITATIVE EXAMPLES: OFT correct, Baseline wrong")
    print("="*70)

    selected = []
    shown = 0
    for idx, (b, o) in enumerate(zip(baseline_results, oft_results)):
        if not b["correct"] and o["correct"] and shown < n:
            selected.append({
                "index": idx,
                "question": b["question"],
                "ground_truth": b["ground_truth"],
                "baseline_prediction": b["prediction"],
                "oft_prediction": o["prediction"],
                "baseline_generated_text": b.get("generated_text", ""),
                "oft_generated_text": o.get("generated_text", ""),
                "baseline_correct": b["correct"],
                "oft_correct": o["correct"],
                "level": b.get("level", "unknown"),
            })
            print(f"\nQuestion: {b['question'][:200]}...")
            print(f"  Ground truth answer : {b['ground_truth']}")
            print(f"  Baseline prediction : {b['prediction']}")
            print(f"  OFT prediction      : {o['prediction']}")
            if show_reasoning:
                print("\n  --- Baseline full reasoning ---")
                print(b.get("generated_text", "").strip() or "<empty>")
                print("\n  --- OFT full reasoning ---")
                print(o.get("generated_text", "").strip() or "<empty>")
            shown += 1

    if shown == 0:
        print("No such examples found in reviewed samples.")

    return selected


def save_qualitative_examples(examples, out_path: str):
    """Save side-by-side qualitative examples with full reasoning text."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"examples": examples}, f, indent=2)
    print(f"Full reasoning qualitative examples saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./output/oft-qwen-gsm8k/final",
                        help="Path to OFT adapter (or base model if --no_adapter)")
    parser.add_argument("--tag", type=str, default="oft",
                        help="Label for this run (used in output filenames)")
    parser.add_argument("--no_adapter", action="store_true",
                        help="Evaluate base model without any adapter (baseline)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both baseline and OFT evaluation and compare")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit to N test examples (default: full 1319)")
    parser.add_argument("--qualitative_examples", type=int, default=5,
                        help="Number of qualitative examples to print/save in compare mode")
    parser.add_argument("--show_reasoning", action="store_true",
                        help="Print full generated reasoning for qualitative examples")
    parser.add_argument("--save_qualitative_path", type=str,
                        default="./output/results/qualitative_examples_full_reasoning.json",
                        help="Path to save qualitative examples with full reasoning")
    args = parser.parse_args()

    if args.compare:
        print("\n--- Evaluating BASELINE (no fine-tuning) ---")
        base_acc, base_results = run_eval(
            MODEL_ID, use_adapter=False, tag="baseline", num_samples=args.num_samples
        )
        print("\n--- Evaluating OFT fine-tuned model ---")
        oft_acc, oft_results = run_eval(
            "./output/oft-qwen-gsm8k/final", use_adapter=True, tag="oft",
            num_samples=args.num_samples
        )
        print(f"\nSummary:")
        print(f"  Baseline accuracy : {base_acc*100:.2f}%")
        print(f"  OFT accuracy      : {oft_acc*100:.2f}%")
        print(f"  Improvement       : +{(oft_acc - base_acc)*100:.2f}%")
        examples = print_qualitative(
            base_results,
            oft_results,
            n=args.qualitative_examples,
            show_reasoning=args.show_reasoning,
        )
        save_qualitative_examples(examples, args.save_qualitative_path)
    else:
        run_eval(
            args.model_path,
            use_adapter=not args.no_adapter,
            tag=args.tag,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
