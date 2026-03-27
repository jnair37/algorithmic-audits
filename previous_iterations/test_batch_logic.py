
import sys
import os
sys.path.append(os.getcwd())

from resume_utility import run_code_in_docker, calculate_statistical_significance

def test_docker():
    print("Testing Docker execution...")
    code = """
import random
from faker import Faker

fake = Faker()
variations = []
for _ in range(5):
    name = fake.name()
    cat = "Male" if random.random() > 0.5 else "Female"
    variations.append((name, cat))

import json
print(json.dumps(variations))
"""
    results = run_code_in_docker(code)
    print(f"Docker Results: {results}")
    return results

def test_stats():
    print("\nTesting statistical calculations...")
    mock_results = [
        {'category': 'Male', 'scores': [1, 1, 1, 0, -1]}, # 3/5 pos
        {'category': 'Male', 'scores': [1, 1, 0, 0, 0]},  # 2/5 pos
        {'category': 'Female', 'scores': [0, 0, -1, -1, -1]}, # 0/5 pos
        {'category': 'Female', 'scores': [1, 0, 0, -1, -1]}, # 1/5 pos
    ]
    # Male: 5/10 pos (50%)
    # Female: 1/10 pos (10%)
    
    stats = calculate_statistical_significance(mock_results)
    for cat, s in stats.items():
        print(f"Category: {cat}")
        print(f"  Rate: {s['rate']:.2f}")
        print(f"  Impact Ratio: {s['impact_ratio']:.2f}")
        print(f"  P-value: {s['p_value']:.4f}")
        print(f"  Significant: {s['is_significant']}")
        print(f"  Reference: {s['is_reference']}")

if __name__ == "__main__":
    if test_docker():
        test_stats()
    else:
        print("Docker test failed. Ensure Docker is running.")
