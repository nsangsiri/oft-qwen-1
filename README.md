# OFT Fine-tuning of Qwen2.5-1.5B on GSM8K

Fine-tuning **Qwen2.5-1.5B** with **Orthogonal Fine-Tuning (OFT)** on the GSM8K math reasoning benchmark.

## Setup

```bash
# Clone repository
git clone https://github.com/nsangsiri/oft-qwen-1.git
cd oft-qwen-1

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

## Evaluation

```bash
# Run both and print comparison + qualitative examples
python evaluate.py --compare

# Quick test on 100 samples
python evaluate.py --compare --num_samples 100
```

Produces JSON results in `./output/results/`:
- `baseline_results.json` — baseline model accuracy & predictions per sample
- `oft_results.json` — OFT fine-tuned model accuracy & predictions per sample
  
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
