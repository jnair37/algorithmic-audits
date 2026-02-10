"""
Customizable Prompt Templates for Synthetic Data Generation using LLMs

This file contains boilerplate prompts that can be adjusted based on user instructions
to generate synthetic variations of resume tokens using Meta Llama or other open-source LLMs.

Usage:
    1. Identify the category of data you want to generate
    2. Customize the prompt template for your specific needs
    3. The LLM will generate comma-separated variations

Model Recommendations:
    - meta-llama/Llama-3.2-3B-Instruct (lightweight, fast)
    - meta-llama/Llama-3.1-8B-Instruct (balanced)
    - meta-llama/Llama-3.1-70B-Instruct (high quality, slower)
"""

# ==============================================================================
# PROMPT TEMPLATE CATALOG
# ==============================================================================

PROMPT_TEMPLATES = {
    
    # --------------------------------------------------------------------------
    # DEMOGRAPHIC DATA (for fairness testing)
    # --------------------------------------------------------------------------
    
    "male_names": """Generate {num} common male first names from diverse cultural backgrounds. Include names representing:
- American/Anglo names (e.g., John, Michael, David)
- African American names (e.g., Jamal, Tyrone, DeShawn)
- Hispanic/Latino names (e.g., Carlos, Juan, Miguel)
- East Asian names (e.g., Wei, Jin, Hiroshi)
- South Asian names (e.g., Rajesh, Arjun, Vikram)
- Middle Eastern names (e.g., Ahmed, Mohammed, Omar)
- European names (e.g., Pierre, Marco, Klaus)

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Michael, Jamal, Carlos, Wei, Ahmed, David, Tyrone, Juan, Jin, Mohammed""",

    "female_names": """Generate {num} common female first names from diverse cultural backgrounds. Include names representing:
- American/Anglo names (e.g., Sarah, Emily, Jennifer)
- African American names (e.g., Latoya, Keisha, Imani)
- Hispanic/Latino names (e.g., Maria, Sofia, Carmen)
- East Asian names (e.g., Mei, Yuki, Li)
- South Asian names (e.g., Priya, Lakshmi, Anjali)
- Middle Eastern names (e.g., Fatima, Aisha, Layla)
- European names (e.g., Sophie, Francesca, Anna)

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Sarah, Latoya, Maria, Mei, Fatima, Emily, Keisha, Sofia, Yuki, Aisha""",

    "last_names": """Generate {num} common last names from diverse cultural backgrounds. Include surnames representing:
- Anglo/American surnames (e.g., Smith, Johnson, Williams)
- African American surnames (e.g., Washington, Jackson, Robinson)
- Hispanic/Latino surnames (e.g., Garcia, Rodriguez, Martinez)
- East Asian surnames (e.g., Wang, Kim, Chen)
- South Asian surnames (e.g., Patel, Singh, Kumar)
- Middle Eastern surnames (e.g., Ali, Hassan, Abdullah)
- European surnames (e.g., Mueller, Dubois, Rossi)

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Smith, Washington, Garcia, Wang, Patel, Ali, Johnson, Jackson, Rodriguez, Kim""",
    
    # --------------------------------------------------------------------------
    # EDUCATIONAL INSTITUTIONS
    # --------------------------------------------------------------------------
    
    "universities_prestigious": """Generate {num} prestigious university names from around the world. Include a mix of:
- US Ivy League and top-tier (e.g., Harvard, Stanford, MIT)
- US public research universities (e.g., Berkeley, Michigan, UCLA)
- UK universities (e.g., Oxford, Cambridge, Imperial College)
- European universities (e.g., ETH Zurich, Sorbonne)
- Asian universities (e.g., Tsinghua, Tokyo, NUS)

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Harvard, Stanford, Oxford, MIT, Cambridge, Berkeley, ETH Zurich, Tsinghua, Yale, Imperial College""",

    "universities_state": """Generate {num} US state university names. Focus on well-known public universities.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: University of Michigan, UC Berkeley, University of Texas, Penn State, Ohio State, University of Washington, University of Florida, UCLA, UNC Chapel Hill, University of Wisconsin""",

    "universities_regional": """Generate {num} regional or less selective US university names.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Central State University, Northern Regional College, Midwest State University, Southern Technical Institute, Eastern Community College""",
    
    # --------------------------------------------------------------------------
    # COMPANIES AND ORGANIZATIONS
    # --------------------------------------------------------------------------
    
    "tech_companies_large": """Generate {num} large, well-known technology companies. Include:
- FAANG/MAANG companies
- Major cloud/enterprise tech companies
- Consumer tech companies
- Hardware manufacturers

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Google, Amazon, Microsoft, Apple, Meta, Netflix, Tesla, Nvidia, Adobe, Salesforce""",

    "tech_companies_startup": """Generate {num} realistic startup or mid-size technology company names. Create plausible company names that sound like tech startups.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: DataFlow Systems, CloudSync Technologies, NexGen Analytics, QuantumLeap AI, StreamVision Labs""",

    "consulting_firms": """Generate {num} consulting firm names, including major management consulting firms and boutique firms.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: McKinsey, BCG, Bain, Deloitte, Accenture, PwC, EY, KPMG, Oliver Wyman, Booz Allen""",

    "finance_companies": """Generate {num} financial services companies including investment banks, asset managers, and fintech companies.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Goldman Sachs, JPMorgan, Morgan Stanley, BlackRock, Citadel, Bridgewater, Fidelity, Vanguard, Charles Schwab, Stripe""",
    
    # --------------------------------------------------------------------------
    # TECHNICAL SKILLS
    # --------------------------------------------------------------------------
    
    "programming_languages": """Generate {num} popular programming languages used in professional software development.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Python, Java, JavaScript, C++, Go, Rust, TypeScript, Swift, Kotlin, Ruby""",

    "web_frameworks": """Generate {num} popular web development frameworks and libraries.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: React, Angular, Vue, Django, Flask, Express, Spring Boot, Ruby on Rails, ASP.NET, Next.js""",

    "databases": """Generate {num} database systems including SQL and NoSQL databases.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: PostgreSQL, MySQL, MongoDB, Redis, Cassandra, Oracle, SQL Server, DynamoDB, Elasticsearch, Neo4j""",

    "cloud_platforms": """Generate {num} cloud computing platforms and services.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: AWS, Google Cloud, Azure, Oracle Cloud, IBM Cloud, DigitalOcean, Heroku, Linode, Cloudflare, Vercel""",
    
    # --------------------------------------------------------------------------
    # JOB TITLES AND ROLES
    # --------------------------------------------------------------------------
    
    "engineering_titles": """Generate {num} software engineering job titles at various levels.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Software Engineer, Senior Engineer, Staff Engineer, Principal Engineer, Engineering Manager, Tech Lead, Developer, Programmer, Architect, DevOps Engineer""",

    "data_titles": """Generate {num} data-related job titles.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Data Scientist, Data Analyst, Machine Learning Engineer, Data Engineer, Research Scientist, AI Engineer, Analytics Manager, Business Intelligence Analyst, Statistician, Quantitative Analyst""",

    "management_titles": """Generate {num} management and leadership job titles.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Manager, Senior Manager, Director, Senior Director, Vice President, Chief Technology Officer, Team Lead, Department Head, Executive Director, General Manager""",

    "business_titles": """Generate {num} business-focused job titles.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Business Analyst, Product Manager, Project Manager, Consultant, Strategy Analyst, Operations Manager, Account Manager, Sales Engineer, Marketing Manager, Financial Analyst""",
    
    # --------------------------------------------------------------------------
    # TEMPORAL DATA
    # --------------------------------------------------------------------------
    
    "years_recent": """Generate {num} years from the past decade (2015-2025). Include a realistic spread.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: 2018, 2019, 2020, 2021, 2022, 2023, 2024""",

    "years_around": """Generate {num} years within 2-3 years of {reference_year}. Center around the reference year.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example (if reference is 2020): 2018, 2019, 2020, 2021, 2022""",

    "durations": """Generate {num} realistic job duration phrases in years.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: 6 months, 1 year, 2 years, 3 years, 5 years, 8 years, 10+ years""",
    
    # --------------------------------------------------------------------------
    # QUALIFICATIONS AND ACHIEVEMENTS
    # --------------------------------------------------------------------------
    
    "degrees": """Generate {num} academic degree types.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Bachelor of Science, Master of Science, PhD, MBA, Bachelor of Arts, Master of Engineering, Associate Degree, Professional Certificate, Doctorate, Bachelor of Engineering""",

    "majors_technical": """Generate {num} technical/STEM academic majors.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Computer Science, Electrical Engineering, Mechanical Engineering, Mathematics, Physics, Statistics, Data Science, Chemical Engineering, Bioengineering, Applied Mathematics""",

    "majors_business": """Generate {num} business and social science academic majors.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: Business Administration, Economics, Finance, Marketing, Management, Accounting, Psychology, Sociology, Political Science, Communications""",

    "certifications": """Generate {num} professional certifications relevant to technology and business.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: AWS Certified Solutions Architect, PMP, CPA, CFA, Certified Scrum Master, Google Cloud Professional, CISSP, Six Sigma Black Belt, CompTIA Security+, Microsoft Certified Azure""",
    
    # --------------------------------------------------------------------------
    # GEOGRAPHIC LOCATIONS
    # --------------------------------------------------------------------------
    
    "us_cities_major": """Generate {num} major US cities where tech jobs are common.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: San Francisco, New York, Seattle, Boston, Austin, Los Angeles, Chicago, Washington DC, Denver, Atlanta""",

    "international_cities": """Generate {num} major international cities with significant tech industries.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: London, Berlin, Toronto, Singapore, Tokyo, Bangalore, Tel Aviv, Sydney, Amsterdam, Dublin""",

    "us_states": """Generate {num} US states with significant tech industries or business activity.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: California, New York, Texas, Washington, Massachusetts, Illinois, Colorado, North Carolina, Georgia, Florida""",
    
}

# ==============================================================================
# HELPER FUNCTIONS FOR PROMPT CUSTOMIZATION
# ==============================================================================

def get_prompt_for_category(category, num_variations=5, **kwargs):
    """
    Get a customized prompt for a specific category.
    
    Args:
        category: Category key from PROMPT_TEMPLATES
        num_variations: Number of variations to generate
        **kwargs: Additional parameters for prompt formatting (e.g., reference_year)
    
    Returns:
        Formatted prompt string
    """
    if category not in PROMPT_TEMPLATES:
        return get_generic_prompt(num_variations)
    
    template = PROMPT_TEMPLATES[category]
    
    # Format with provided parameters
    try:
        return template.format(num=num_variations, **kwargs)
    except KeyError as e:
        print(f"Warning: Missing parameter {e} for category {category}")
        return template.format(num=num_variations)


def get_generic_prompt(token, num_variations=5):
    """
    Get a generic prompt for tokens not in predefined categories.
    
    Args:
        token: The original token
        num_variations: Number of variations to generate
    
    Returns:
        Generic prompt string
    """
    return f"""Generate {num_variations} similar or alternative terms to "{token}" that could be used in the same professional/resume context. Maintain similar formality and specificity.

Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example: [provide variations of {token}]"""


def create_custom_prompt(instruction, num_variations=5):
    """
    Create a custom prompt based on user instructions.
    
    Args:
        instruction: User's custom instruction for what to generate
        num_variations: Number of variations to generate
    
    Returns:
        Custom prompt string
    """
    return f"""{instruction}

Generate exactly {num_variations} items. Respond ONLY with a comma-separated list, no explanations, numbering, or formatting.

Example format: Item1, Item2, Item3, Item4, Item5"""


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    # Example 1: Get prompt for male names
    prompt1 = get_prompt_for_category("male_names", num_variations=25)
    print("Example 1 - Male Names Prompt:")
    print(prompt1)
    print("\n" + "="*80 + "\n")
    
    # Example 2: Get prompt for years around 2020
    prompt2 = get_prompt_for_category("years_around", num_variations=7, reference_year=2020)
    print("Example 2 - Years Around 2020 Prompt:")
    print(prompt2)
    print("\n" + "="*80 + "\n")
    
    # Example 3: Custom prompt
    custom_instruction = "Generate diverse names of programming bootcamps and coding academies"
    prompt3 = create_custom_prompt(custom_instruction, num_variations=10)
    print("Example 3 - Custom Bootcamp Names Prompt:")
    print(prompt3)
    print("\n" + "="*80 + "\n")