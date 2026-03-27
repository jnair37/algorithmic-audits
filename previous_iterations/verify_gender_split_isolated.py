from faker import Faker

def generate_names_faker(num, dimension='Gender'):
    """
    COPIED LOGIC FROM resume_utility.py for verification
    """
    unique_names = {}
    
    if dimension == 'Gender':
        f = Faker('en_US')
        target_male = num // 2
        target_female = num - target_male
        
        # Generate male names
        for _ in range(target_male * 5): # Allow for duplicates
            if len(unique_names) >= target_male: break
            name = f.first_name_male()
            if name not in unique_names:
                unique_names[name] = 'Male'
        
        # Generate female names
        for _ in range(num * 5): # Allow for duplicates
            if len(unique_names) >= num: break
            name = f.first_name_female()
            if name not in unique_names:
                unique_names[name] = 'Female'
    
    return list(unique_names.items())[:num]

def test_gender_split():
    print("Testing Gender Split Logic (Isolated)...")
    
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
    print("Test Case 1 (Even) PASSED")
    
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
    # 11 // 2 = 5 male, 11 - 5 = 6 female (as per current logic)
    assert male_count == 5
    assert female_count == 6
    print("Test Case 2 (Odd) PASSED")
    
    print("\nGender split verification PASSED!")

if __name__ == "__main__":
    test_gender_split()
