from faker import Faker

def generate_names_faker(num, gender='Random', locale='English (US)'):
    """
    Copy of the logic implemented in resume_utility.py for isolated testing.
    """
    locale_map = {
        'English (US)': 'en_US',
        'Spanish (Spain)': 'es_ES',
        'Chinese (China)': 'zh_CN',
        'Hindi (India)': 'hi_IN',
        'French (France)': 'fr_FR',
        'German (Germany)': 'de_DE',
        'Japanese (Japan)': 'ja_JP'
    }
    
    selected_locale = locale_map.get(locale, 'en_US')
    f = Faker(selected_locale)
    
    names = []
    for _ in range(num):
        if gender == 'Male':
            names.append(f.first_name_male())
        elif gender == 'Female':
            names.append(f.first_name_female())
        else:
            names.append(f.first_name())
    
    return list(set(names))

def test_isolated_faker():
    print("Testing Isolated Faker logic...")
    
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
    
    print("\nAll isolated tests passed!")

if __name__ == "__main__":
    test_isolated_faker()
