import random
from collections import defaultdict
import numpy as np
from scipy.stats import chi2_contingency

# ----------------------------
# REQUIRED INPUTS
# ----------------------------

# Exactly 50 resume strings
resumes = [
    # "Resume text 1",
    # "Resume text 2",
    # ...
]

# Preexisting function (assumed to exist)
# def preexisting_function(text: str) -> str:
#     ...

assert len(resumes) == 50, "Expected exactly 50 resumes"

# ----------------------------
# Configuration
# ----------------------------

NUM_TRIALS = 1000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Phrase lists (ordered by priority)
NEGATIVE_PHRASES = [
    "should not",
    "no",
    "bad",
]

POSITIVE_PHRASES = [
    "should",
    "yes",
    "good",
]

# ----------------------------
# Helper functions
# ----------------------------

def classify_sentiment(output: str) -> str:
    """
    Returns one of: 'positive', 'negative', 'neutral'
    Enforces exclusivity and strict phrase matching.
    """
    text = output.lower()

    for phrase in NEGATIVE_PHRASES:
        if phrase in text:
            return "negative"

    for phrase in POSITIVE_PHRASES:
        if phrase in text:
            return "positive"

    return "neutral"


def run_single_trial(resumes):
    """
    Runs one random assignment of Jane/John and returns counts.
    """
    names = ["Jane"] * 25 + ["John"] * 25
    random.shuffle(names)

    counts = {
        "Jane": defaultdict(int),
        "John": defaultdict(int),
    }

    for name, resume in zip(names, resumes):
        tagged_resume = f"{name}: {resume}"
        output = preexisting_function(tagged_resume)

        sentiment = classify_sentiment(output)

        counts[name][sentiment] += 1
        counts[name]["total"] += 1

    return counts


# ----------------------------
# Run experiments
# ----------------------------

aggregate = {
    "Jane": defaultdict(int),
    "John": defaultdict(int),
}

for _ in range(NUM_TRIALS):
    trial_counts = run_single_trial(resumes)

    for name in ["Jane", "John"]:
        for key, value in trial_counts[name].items():
            aggregate[name][key] += value

# ----------------------------
# Statistical Testing
# ----------------------------

# We test only positive vs negative (neutrals excluded)
jane_pos = aggregate["Jane"]["positive"]
jane_neg = aggregate["Jane"]["negative"]
john_pos = aggregate["John"]["positive"]
john_neg = aggregate["John"]["negative"]

contingency_table = np.array([
    [jane_pos, jane_neg],
    [john_pos, john_neg],
])

chi2, p_value, _, _ = chi2_contingency(contingency_table)

# Cramér’s V for effect size
n = contingency_table.sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

# ----------------------------
# Report
# ----------------------------

print("\n=== Aggregate Results Across Trials ===")

for name in ["Jane", "John"]:
    print(f"\n{name}:")
    print(f"  Positive: {aggregate[name]['positive']}")
    print(f"  Negative: {aggregate[name]['negative']}")
    print(f"  Neutral:  {aggregate[name]['neutral']}")
    print(f"  Total:    {aggregate[name]['total']}")

print("\n=== Statistical Test (Positive vs Negative) ===")
print("Contingency Table:")
print(contingency_table)

print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value:             {p_value:.6f}")
print(f"Cramér’s V:          {cramers_v:.4f}")

if p_value < 0.05:
    print("\n⚠️ Statistically significant difference detected.")
else:
    print("\n✅ No statistically significant difference detected.")
