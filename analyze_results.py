"""
Comprehensive analysis of OFT vs Baseline results on GSM8K.

Answers three key questions:
1. Did OFT preserve knowledge better than expected?
2. Was the improvement consistent across problem types?
3. Any surprising results?
"""

import json
import os
from collections import defaultdict


def load_results(name: str) -> list:
    """Load evaluation results from JSON."""
    path = f"./output/results/{name}_results.json"
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run: python evaluate.py --compare")
        exit(1)
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def analyze_knowledge_preservation(baseline: list, oft: list):
    """Q1: Did OFT preserve knowledge?"""
    print("\n" + "=" * 80)
    print("1. KNOWLEDGE PRESERVATION ANALYSIS")
    print("=" * 80)
    
    both_correct = sum(1 for b, o in zip(baseline, oft) if b["correct"] and o["correct"])
    both_wrong = sum(1 for b, o in zip(baseline, oft) if not b["correct"] and not o["correct"])
    baseline_only = sum(1 for b, o in zip(baseline, oft) if b["correct"] and not o["correct"])
    oft_only = sum(1 for b, o in zip(baseline, oft) if not b["correct"] and o["correct"])
    total = len(baseline)
    
    baseline_correct_count = sum(1 for b in baseline if b["correct"])
    oft_correct_count = sum(1 for o in oft if o["correct"])
    
    print(f"\nConfusion Matrix (out of {total} samples):")
    print(f"  Both correct:                    {both_correct:4d} ({100*both_correct/total:5.1f}%)")
    print(f"  Both wrong:                      {both_wrong:4d} ({100*both_wrong/total:5.1f}%)")
    print(f"  Baseline correct, OFT wrong:     {baseline_only:4d} ({100*baseline_only/total:5.1f}%) ⚠️  FORGETTING")
    print(f"  Baseline wrong, OFT correct:     {oft_only:4d} ({100*oft_only/total:5.1f}%) ✓ IMPROVEMENT")
    
    baseline_acc = baseline_correct_count / total
    oft_acc = oft_correct_count / total
    
    # Knowledge preservation: of what baseline got right, what % did OFT also get right?
    if baseline_correct_count > 0:
        preservation_rate = both_correct / baseline_correct_count
        print(f"\nKnowledge Preservation Rate: {preservation_rate*100:.1f}%")
        print(f"  → Out of {baseline_correct_count} problems baseline solved,")
        print(f"    OFT maintained {both_correct} ({preservation_rate*100:.1f}%) of those correct answers")
    
    print(f"\nAccuracy Summary:")
    print(f"  Baseline: {baseline_acc*100:.2f}% ({baseline_correct_count}/{total})")
    print(f"  OFT:      {oft_acc*100:.2f}% ({oft_correct_count}/{total})")
    print(f"  Gain:     +{(oft_acc - baseline_acc)*100:.2f} percentage points")
    
    # Interpretation
    if baseline_only > 0 and baseline_only / total > 0.05:
        print(f"\n⚠️  WARNING: OFT forgot {baseline_only} answers baseline got right.")
        print(f"   This suggests potential catastrophic forgetting.")
    else:
        print(f"\n✓ GOOD: Minimal catastrophic forgetting detected ({baseline_only} cases).")
    
    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_only": baseline_only,
        "oft_only": oft_only,
        "preservation_rate": preservation_rate if baseline_correct_count > 0 else 0,
        "baseline_acc": baseline_acc,
        "oft_acc": oft_acc,
    }


def analyze_by_difficulty(baseline: list, oft: list):
    """Q2: Consistency across problem types (difficulty levels)."""
    print("\n" + "=" * 80)
    print("2. CONSISTENCY ANALYSIS BY DIFFICULTY LEVEL")
    print("=" * 80)
    
    by_level = defaultdict(lambda: {"baseline": [], "oft": []})
    
    for b, o in zip(baseline, oft):
        level = b.get("level", "unknown")
        by_level[level]["baseline"].append(b["correct"])
        by_level[level]["oft"].append(o["correct"])
    
    if not by_level:
        print("\n⚠️  No difficulty levels found in data.")
        return {}
    
    print(f"\n{'Level':<12} {'Baseline':<12} {'OFT':<12} {'Improvement':<12} {'Count':<8}")
    print("-" * 60)
    
    stats = {}
    for level in sorted(by_level.keys()):
        baseline_results = by_level[level]["baseline"]
        oft_results = by_level[level]["oft"]
        
        baseline_acc = sum(baseline_results) / len(baseline_results) if baseline_results else 0
        oft_acc = sum(oft_results) / len(oft_results) if oft_results else 0
        improvement = (oft_acc - baseline_acc) * 100
        
        print(f"{level:<12} {baseline_acc*100:>6.1f}%{'':<4} {oft_acc*100:>6.1f}%{'':<4} {improvement:>+6.1f}%{'':<4} {len(baseline_results):>6d}")
        
        stats[level] = {
            "baseline_acc": baseline_acc,
            "oft_acc": oft_acc,
            "improvement": improvement,
            "count": len(baseline_results),
        }
    
    # Check consistency
    improvements = [s["improvement"] for s in stats.values()]
    if len(improvements) > 1:
        max_imp = max(improvements)
        min_imp = min(improvements)
        variance = max_imp - min_imp
        
        print(f"\nConsistency Metric:")
        print(f"  Max improvement: {max_imp:+.1f}%")
        print(f"  Min improvement: {min_imp:+.1f}%")
        print(f"  Variance: {variance:.1f}%")
        
        if variance > 15:
            print(f"  → Improvement is INCONSISTENT across difficulty levels")
        else:
            print(f"  → Improvement is relatively CONSISTENT across difficulty levels")
    
    return stats


def analyze_surprising_results(baseline: list, oft: list):
    """Q3: Surprising results and edge cases."""
    print("\n" + "=" * 80)
    print("3. SURPRISING RESULTS & EDGE CASES")
    print("=" * 80)
    
    # Cases where baseline correct but OFT wrong (forgetting)
    forgetting_cases = [(i, b, o) for i, (b, o) in enumerate(zip(baseline, oft)) 
                        if b["correct"] and not o["correct"]]
    
    print(f"\n📌 FORGETTING CASES (Baseline correct, OFT wrong): {len(forgetting_cases)} cases")
    if forgetting_cases:
        print("\n   Examples:")
        for i, b, o in forgetting_cases[:3]:
            print(f"\n   Example {i+1}:")
            print(f"   Question: {b['question'][:80]}...")
            print(f"   Ground Truth: {b['ground_truth']}")
            print(f"   Baseline pred: {b['prediction']} ✓")
            print(f"   OFT pred:      {o['prediction']} ✗")
            print("   Baseline full reasoning:")
            print((b.get("generated_text", "") or "").strip()[:1200] or "<empty>")
            print("   OFT full reasoning:")
            print((o.get("generated_text", "") or "").strip()[:1200] or "<empty>")
    
    # Cases where OFT correct but baseline wrong (improvement)
    improvement_cases = [(i, b, o) for i, (b, o) in enumerate(zip(baseline, oft)) 
                         if not b["correct"] and o["correct"]]
    
    print(f"\n📌 IMPROVEMENT CASES (Baseline wrong, OFT correct): {len(improvement_cases)} cases")
    if improvement_cases:
        print("\n   Examples:")
        for i, b, o in improvement_cases[:3]:
            print(f"\n   Example {i+1}:")
            print(f"   Question: {b['question'][:80]}...")
            print(f"   Ground Truth: {b['ground_truth']}")
            print(f"   Baseline pred: {b['prediction']} ✗")
            print(f"   OFT pred:      {o['prediction']} ✓")
            print("   Baseline full reasoning:")
            print((b.get("generated_text", "") or "").strip()[:1200] or "<empty>")
            print("   OFT full reasoning:")
            print((o.get("generated_text", "") or "").strip()[:1200] or "<empty>")
    
    # Ratio analysis
    total = len(baseline)
    if total > 0:
        forgetting_ratio = len(forgetting_cases) / total
        improvement_ratio = len(improvement_cases) / total
        
        print(f"\n📊 Ratio Analysis (out of {total} samples):")
        print(f"   Forgetting ratio: {forgetting_ratio*100:.2f}% ({len(forgetting_cases)} cases)")
        print(f"   Improvement ratio: {improvement_ratio*100:.2f}% ({len(improvement_cases)} cases)")
        
        if improvement_ratio > forgetting_ratio * 2:
            print(f"   → More improvements than forgetting: Good sign! ✓")
        elif forgetting_ratio > improvement_ratio * 2:
            print(f"   → More forgetting than improvements: Potential issue ⚠️")
    
    return {
        "forgetting_count": len(forgetting_cases),
        "improvement_count": len(improvement_cases),
        "forgetting_cases": forgetting_cases,
        "improvement_cases": improvement_cases,
    }


def generate_summary(preservation_stats, difficulty_stats, surprising_stats):
    """Generate final summary."""
    print("\n" + "=" * 80)
    print("SUMMARY & INTERPRETATION")
    print("=" * 80)
    
    print("\n✅ Key Takeaways:")
    
    # Knowledge preservation
    if preservation_stats["preservation_rate"] > 0.95:
        print(f"  1. OFT preserved {preservation_stats['preservation_rate']*100:.1f}% of baseline knowledge")
        print(f"     → Excellent knowledge preservation!")
    elif preservation_stats["preservation_rate"] > 0.90:
        print(f"  1. OFT preserved {preservation_stats['preservation_rate']*100:.1f}% of baseline knowledge")
        print(f"     → Good knowledge preservation")
    else:
        print(f"  1. OFT preserved {preservation_stats['preservation_rate']*100:.1f}% of baseline knowledge")
        print(f"     → Some forgetting detected")
    
    # Accuracy gain
    gain = (preservation_stats["oft_acc"] - preservation_stats["baseline_acc"]) * 100
    if gain > 10:
        print(f"  2. OFT achieved +{gain:.1f}% accuracy gain → Strong improvement!")
    elif gain > 5:
        print(f"  2. OFT achieved +{gain:.1f}% accuracy gain → Moderate improvement")
    elif gain > 0:
        print(f"  2. OFT achieved +{gain:.1f}% accuracy gain → Slight improvement")
    else:
        print(f"  2. OFT achieved {gain:.1f}% accuracy change → No clear improvement")
    
    # Consistency
    if difficulty_stats:
        improvements = [s["improvement"] for s in difficulty_stats.values()]
        variance = max(improvements) - min(improvements) if improvements else 0
        if variance < 10:
            print(f"  3. Improvement consistent across difficulty levels (variance: {variance:.1f}%)")
        else:
            print(f"  3. Improvement varies by difficulty level (variance: {variance:.1f}%)")
    
    # Ratio
    forgetting = surprising_stats["forgetting_count"]
    improvement = surprising_stats["improvement_count"]
    if improvement > forgetting:
        print(f"  4. More improvements ({improvement}) than forgetting cases ({forgetting}) → Positive!")
    
    print("\n💡 Recommendation for report:")
    print("  → Use preservation rate, accuracy comparison, and qualitative examples")
    print("  → Highlight consistency/inconsistency by difficulty")
    print("  → Show 2-3 concrete before/after examples from improvement_cases")


def main():
    print("Loading evaluation results...")
    baseline = load_results("baseline")
    oft = load_results("oft")
    
    print(f"Loaded {len(baseline)} baseline results and {len(oft)} OFT results")
    
    # Run all analyses
    preservation_stats = analyze_knowledge_preservation(baseline, oft)
    difficulty_stats = analyze_by_difficulty(baseline, oft)
    surprising_stats = analyze_surprising_results(baseline, oft)
    
    # Summary
    generate_summary(preservation_stats, difficulty_stats, surprising_stats)
    
    # Save to file
    output_file = "./output/analysis_report.json"
    os.makedirs("./output", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "preservation": preservation_stats,
            "by_difficulty": difficulty_stats,
            "surprising": {
                "forgetting_count": surprising_stats["forgetting_count"],
                "improvement_count": surprising_stats["improvement_count"],
            }
        }, f, indent=2)
    print(f"\n📄 Analysis saved to {output_file}")


if __name__ == "__main__":
    main()
