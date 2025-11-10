"""
Predefined vulnerabilities and attacks for DeepTeam configuration.
Modify these dictionaries to add, remove, or update available options.
"""

# Dictionary mapping vulnerability types to their sub-types
# If a vulnerability has no sub-types, use an empty list []
VULNERABILITY_TYPES = {
    "Bias": [
        "age",
        "race", 
        "gender",
        "religion",
        "nationality",
        "disability",
        "sexual_orientation"
    ],
    "Misinformation": [
        "financial",
        "medical",
        "political",
        "scientific"
    ],
    "PII": [
        "social_security",
        "credit_card",
        "email",
        "phone",
        "address",
        "passport"
    ],
    "Excessive Agency": [],  # No sub-types
    "Hallucination": [
        "factual",
        "contextual"
    ],
    "Toxicity": [
        "hate_speech",
        "profanity",
        "harassment",
        "violence"
    ],
    "Data Leakage": [
        "training_data",
        "user_data",
        "proprietary_info"
    ],
    "Unformatted": [],  # No sub-types
}

# Dictionary mapping attack categories to their methods
ATTACK_CATEGORIES = {
    "Single Turn": [
        "Prompt Injection",
        "Jailbreaking",
        "Context Poisoning",
        "ROT13",
        "Base64 Encoding",
        "Leetspeak",
        "Character Substitution",
        "Payload Splitting"
    ],
    "Multi Turn": [
        "Crescendo",
        "Contextual Redirection",
        "Progressive Escalation",
        "Gray Box Attack"
    ]
}

