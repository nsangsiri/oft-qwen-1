# OFT Fine-tuning of Qwen2.5-1.5B on GSM8K

Fine-tuning **Qwen2.5-1.5B** with **Orthogonal Fine-Tuning (OFT)** on the GSM8K math reasoning benchmark.

## What is OFT?

Orthogonal Fine-Tuning (OFT) ([Qiu et al., NeurIPS 2023](https://arxiv.org/abs/2306.07280)) is a parameter-efficient fine-tuning (PEFT) method that updates pretrained weights via **multiplicative orthogonal transformations**:

$$W' = R \cdot W, \quad R^\top R = I$$

Unlike LoRA (which adds a low-rank residual $W' = W + AB$), OFT preserves the **hyperspherical energy** of the weight matrix — the pairwise angles between neurons — which prevents catastrophic forgetting and improves generalization.

| Method | Update type | Preserves geometry | Params |
|--------|-------------|-------------------|--------|
| Full fine-tune | Additive | No | 100% |
| LoRA | Additive low-rank | No | ~0.1–1% |
| **OFT** | **Multiplicative orthogonal** | **Yes** | **~0.1–1%** |

## Setup

```bash
# Clone repository
git clone <your-repo-url>
cd oft-qwen-gsm8k

# Create and activate conda environment
conda env create -f environment.yml
conda activate oft-qwen
```

## Repository Structure

```
oft-qwen-gsm8k/
├── train.py          # OFT fine-tuning script
├── evaluate.py       # GSM8K exact-match accuracy evaluation
├── analyze_results.py # Comprehensive analysis (Q&A for report)
├── plot_loss.py      # Plot training/validation loss curves
├── requirements.txt
├── output/
│   ├── oft-qwen-gsm8k/     # Saved model checkpoints
│   ├── training_logs.json  # Loss log (auto-generated)
│   ├── loss_curve.png      # Loss plot (auto-generated)
│   ├── analysis_report.json # Analysis results (auto-generated)
│   └── results/
│       ├── baseline_results.json
│       └── oft_results.json
└── README.md
```

## Training

```bash
python train.py
```

**Key OFT hyperparameters** (edit in `train.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` (rank/block size) | 8 | Controls expressiveness of the orthogonal matrix |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj` | Which layers receive OFT |
| `coft` | `False` | Set `True` for Constrained OFT (Cayley parameterisation) |
| `block_share` | `False` | Share one block across all layers for fewer params |
| Learning rate | 2e-4 | Cosine schedule with 5% warmup |
| Epochs | 3 | |
| Effective batch size | 16 (4 × 4 grad acc) | |

## Evaluation

```bash
# Evaluate the fine-tuned model only
python evaluate.py --model_path ./output/oft-qwen-gsm8k/final --tag oft

# Evaluate baseline (base model without fine-tuning)
python evaluate.py --no_adapter --tag baseline

# Run both and print comparison + qualitative examples
python evaluate.py --compare

# Quick test on 100 samples
python evaluate.py --compare --num_samples 100

# Print full reasoning in terminal and save side-by-side examples to JSON
python evaluate.py --compare --show_reasoning --qualitative_examples 5
```

Produces JSON results in `./output/results/`:
- `baseline_results.json` — baseline model accuracy & predictions per sample
- `oft_results.json` — OFT fine-tuned model accuracy & predictions per sample
- `qualitative_examples_full_reasoning.json` — report-ready side-by-side examples with full reasoning

Note: each sample in `baseline_results.json` and `oft_results.json` contains `generated_text`, which is the full model reasoning output for that question.

## Comprehensive Analysis

After running evaluation, analyze results to answer key report questions:

```bash
python analyze_results.py
```

This answers:
1. **Knowledge Preservation**: Did OFT preserve baseline knowledge?
2. **Consistency**: Was improvement consistent across difficulty levels?
3. **Surprising Results**: Show concrete forgetting & improvement cases

Produces:
- Console output with tables, stats, and qualitative examples
- `./output/analysis_report.json` with structured metrics

To print qualitative examples including full reasoning in the analysis output:

```bash
python analyze_results.py > report_analysis.txt
```

**Key outputs for your report:**
- Knowledge preservation rate (%)
- Accuracy comparison table
- Improvement by difficulty level
- Concrete before/after examples

## Plotting Loss Curves

```bash
python plot_loss.py
```

Produces `./output/loss_curve.png` with training loss (smoothed with EMA) and validation loss per epoch.

## Expected Results

| Model | GSM8K Accuracy |
|-------|---------------|
| Qwen2.5-1.5B (baseline) | ~55–60% |
| Qwen2.5-1.5B + OFT | ~65–72% (estimated) |

*Note: exact numbers depend on hardware, random seed, and full training duration.*

## Experiment Design (for report)

1. **Baseline**: Evaluate `Qwen2.5-1.5B` zero-shot on GSM8K.
2. **OFT**: Fine-tune with `r=8`, then evaluate.
3. **Analysis**: Compare knowledge preservation, consistency across difficulty levels, and qualitative results.
4. **(Optional) LoRA comparison**: Replace `OFTConfig` with `LoraConfig(r=8)` in `train.py` — same target modules, same training budget — for a direct comparison.

## Report Checklist (3 pages)

### Page 1: Introduction & Method
- [ ] What is OFT? (vs. LoRA, full fine-tune)
- [ ] Why GSM8K? (math reasoning benchmark)
- [ ] Hyperparameter table (rank, target modules, learning rate, etc.)
- [ ] Training setup (GPU, batch size, epochs)

### Page 2: Results & Analysis
- [ ] **Training Loss Curve** — Include plot from `plot_loss.py`
- [ ] **Accuracy Comparison Table**:
  ```
  | Model | Accuracy | Improvement |
  |-------|----------|-------------|
  | Baseline | XX% | — |
  | OFT | YY% | +ZZ% |
  ```
- [ ] **Knowledge Preservation** — From `python analyze_results.py`:
  - Preservation rate (%)
  - Forgetting cases (if any)
  
- [ ] **Consistency by Difficulty** — Table showing improvement per level
- [ ] **Qualitative Examples** (2–3 cases):
  - Question
  - Ground truth
  - Baseline prediction (✗ or ✓)
  - OFT prediction (✗ or ✓)

### Page 3: Discussion & Conclusion
- [ ] Key findings (did OFT work as expected?)
- [ ] Comparison to literature (LoRA, other PEFT methods)
- [ ] Why OFT works (preserves geometry, avoids catastrophic forgetting)
- [ ] Limitations (model size, training time, dataset)
- [ ] Future work (larger models, other tasks)

### Quick Start for Report
```bash
# 1. Train and evaluate (if not done already)
python train.py
python evaluate.py --compare

# 2. Generate visualizations
python plot_loss.py
python analyze_results.py > report_analysis.txt

# 3. Use outputs in your report:
# - ./output/loss_curve.png (for Page 2)
# - ./output/results/baseline_results.json (extract examples for qualitative analysis)
# - ./output/results/oft_results.json (extract examples for qualitative analysis)
# - report_analysis.txt (for tables and statistics)
```

## Citation

```bibtex
@inproceedings{qiu2023controlling,
  title     = {Controlling Text-to-Image Diffusion by Orthogonal Fine-tuning},
  author    = {Qiu, Zeju and others},
  booktitle = {NeurIPS},
  year      = {2023}
}

@misc{qwen2.5,
  title  = {Qwen2.5 Technical Report},
  author = {Qwen Team},
  year   = {2024}
}
```
