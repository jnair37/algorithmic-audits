import sys
import os

# Add the current directory to path so we can import resume_utility
sys.path.append(os.getcwd())

from resume_utility import generate_names_faker

def test_faker():
    print("Testing Faker generation...")
    
    # Test 1: Male names, US English
    print("\nTest 1: Male names, English (US)")
    male_names = generate_names_faker(5, gender='Male', locale='English (US)')
    print(f"Generated: {male_names}")
    assert len(male_names) > 0
    
    # Test 2: Female names, Spanish (Spain)
    print("\nTest 2: Female names, Spanish (Spain)")
    spanish_names = generate_names_faker(5, gender='Female', locale='Spanish (Spain)')
    print(f"Generated: {spanish_names}")
    assert len(spanish_names) > 0
    
    # Test 3: Random names, Chinese (China)
    print("\nTest 3: Random names, Chinese (China)")
    chinese_names = generate_names_faker(5, gender='Random', locale='Chinese (China)')
    print(f"Generated: {chinese_names}")
    assert len(chinese_names) > 0
    
    print("\nAll Faker generation tests passed!")

if __name__ == "__main__":
    test_faker()
