from resume_utility import generate_names_faker

def test_gender_split():
    print("Testing Gender Split Logic...")
    
    # Test case 1: Even number
    num_vars = 20
    names = generate_names_faker(num_vars, dimension='Gender')
    
    male_count = sum(1 for _, cat in names if cat == 'Male')
    female_count = sum(1 for _, cat in names if cat == 'Female')
    
    print(f"Requested: {num_vars}")
    print(f"Generated: {len(names)}")
    print(f"Male: {male_count}")
    print(f"Female: {female_count}")
    
    assert len(names) == num_vars
    assert male_count == 10
    assert female_count == 10
    
    # Test case 2: Odd number
    num_vars = 11
    names = generate_names_faker(num_vars, dimension='Gender')
    
    male_count = sum(1 for _, cat in names if cat == 'Male')
    female_count = sum(1 for _, cat in names if cat == 'Female')
    
    print(f"\nRequested: {num_vars}")
    print(f"Generated: {len(names)}")
    print(f"Male: {male_count}")
    print(f"Female: {female_count}")
    
    assert len(names) == num_vars
    # 11 // 2 = 5 male, 11 - 5 = 6 female
    assert male_count == 5
    assert female_count == 6
    
    print("\nGender split verification PASSED!")

if __name__ == "__main__":
    test_gender_split()
