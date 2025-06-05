# filter_ultrafeedback_utils.py

from datasets import Dataset
from typing import List

# Define keyword categories
PROGRAMMING_KEYWORDS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "rust", "go", "swift", "kotlin",
    "program", "programming", "write code", "script", "source code", "software", "compiler",
    "debug", "function", "method", "class", "variable", "loop", "recursion",
    "algorithm", "complexity", "data structure", "hash map", "array", "linked list", "binary tree"
]

MATH_KEYWORDS = [
    "math", "mathematics", "equation", "expression", "formula", "solve", "proof",
    "calculus", "algebra", "geometry", "trigonometry", "linear algebra",
    "integral", "derivative", "limit", "matrix", "vector", "function", "domain", "range"
]

STEM_KEYWORDS = [
    "physics", "force", "energy", "speed", "acceleration", "distance",
    "chemistry", "reaction", "compound", "molecule",
    "statistics", "probability", "mean", "median", "distribution",
    "compute", "calculate", "evaluate", "numeric", "scientific", "logical reasoning", "truth table"
]

# Combine all keywords
ALL_CODE_STEM_KEYWORDS = list(set(PROGRAMMING_KEYWORDS + MATH_KEYWORDS + STEM_KEYWORDS))

def is_code_stem_prompt(prompt: str, keywords: List[str] = ALL_CODE_STEM_KEYWORDS) -> bool:
    """Check if the prompt contains any code/STEM-related keywords."""
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in keywords)

def filter_code_stem_dpo(dataset: Dataset, min_rating: float = 4.0, min_gap: float = 0.5) -> Dataset:
    """Filter a HuggingFace dataset to retain only high-quality code/STEM prompts."""
    
    def keep(example):
        prompt = example.get("prompt", "")
        rating = example.get("rating", 0.0)
        rejected_rating = example.get("rejected_rating", 0.0)
        gap = rating - rejected_rating

        return (
            is_code_stem_prompt(prompt)
            and rating >= min_rating
            and gap >= min_gap
        )

    return dataset.filter(keep)