import random
from datasets import load_dataset, Dataset

MATH_SUBJECTS = [
    "algebra",
    "geometry",
    "number_theory",
    "counting_and_probability",
    "prealgebra",
    "precalculus",
    "intermediate_algebra"
]

def format_dataset_dpo_style(dataset, dataset_name):
    formatted = []

    if dataset_name == "sciq":
        for ex in dataset:
            formatted.append({
                "instruction": ex["question"],
                "output": ex["correct_answer"],
                "dataset": dataset_name
            })

    elif dataset_name == "openbookqa_main":
        for ex in dataset:
            q = ex["question_stem"]
            choices = ex["choices"]["text"]
            if "answerKey" in ex:
                ans = choices[ord(ex["answerKey"]) - ord("A")]
                instruction = f"{q} Here are the options: {', '.join(choices)}. Which one is correct?"
                formatted.append({
                    "instruction": instruction,
                    "output": ans,
                    "dataset": dataset_name
                })

    elif dataset_name.startswith("hendrycks_math_"):
        for ex in dataset:
            formatted.append({
                "instruction": f"Solve this math problem:\n\n{ex['problem']}",
                "output": ex["solution"],
                "dataset": dataset_name
            })

    elif dataset_name == "mbpp":
        for ex in dataset:
            formatted.append({
                "instruction": f"Write a Python function that does the following:\n\n{ex['text']}",
                "output": ex["code"],
                "dataset": dataset_name
            })

    elif dataset_name == "MathInstruct":
        for ex in dataset:
            formatted.append({
                "instruction": ex["instruction"].strip(),
                "output": ex["output"].strip(),
                "dataset": dataset_name
            })

    return formatted

# Dataset combinations
COMBOS = {
    "balanced": {
        "sciq": ("sciq", None, 10000),
        "openbookqa_main": ("openbookqa", "main", 10000),
        "mbpp": ("mbpp", None, 10000),
        "hendrycks_math_all": ("EleutherAI/hendrycks_math", None, 14000),  # 2k x 7
        "MathInstruct": ("TIGER-Lab/MathInstruct", None, 20000),
    },
    "math_only": {
        "hendrycks_math_all": ("EleutherAI/hendrycks_math", None, 14000),
        "MathInstruct": ("TIGER-Lab/MathInstruct", None, 20000),
    },
    "balanced_plus": {
    "sciq": ("sciq", None, 8000),
    "openbookqa_main": ("openbookqa", "main", 8000),
    "mbpp": ("mbpp", None, 12000),
    "MathInstruct": ("TIGER-Lab/MathInstruct", None, 20000),
    "hendrycks_math_all": ("EleutherAI/hendrycks_math", None, 20000),
    },
    "curriculum": {
    "sciq": ("sciq", None, 4000),
    "mbpp": ("mbpp", None, 4000),
    "openbookqa_main": ("openbookqa", "main", 4000),
    "MathInstruct": ("TIGER-Lab/MathInstruct", None, 16000),
    "hendrycks_math_all": ("EleutherAI/hendrycks_math", None, 12000),
    },
    "code_only": {
        "mbpp": ("mbpp", None, 15000),
    },
    "lightweight": {
        "sciq": ("sciq", None, 5000),
        "mbpp": ("mbpp", None, 5000),
        "MathInstruct": ("TIGER-Lab/MathInstruct", None, 10000),
    },
    "reasoning": {
        "openbookqa_main": ("openbookqa", "main", 10000),
        "sciq": ("sciq", None, 10000),
    }
}

def load_hendrycks_all(target_size):
    total = []
    per_subject = target_size // len(MATH_SUBJECTS)
    for subject in MATH_SUBJECTS:
        print(f"📦 Loading hendrycks_math ({subject}) ...")
        raw = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        formatted = format_dataset_dpo_style(raw, f"hendrycks_math_{subject}")
        if len(formatted) > per_subject:
            formatted = random.sample(formatted, per_subject)
        total.extend(formatted)
    return total

def build_combo(combo_name):
    assert combo_name in COMBOS, f"Invalid combo name: {combo_name}"
    sources = COMBOS[combo_name]
    all_data = []

    for ds_key, (path, subset, target_size) in sources.items():
        if ds_key == "hendrycks_math_all":
            all_data.extend(load_hendrycks_all(target_size))
        else:
            print(f"\n📦 Loading {ds_key} ...")
            raw = load_dataset(path, subset, split="train")
            formatted = format_dataset_dpo_style(raw, ds_key)
            print(f"✅ Formatted: {len(formatted)}")

            if len(formatted) > target_size:
                formatted = random.sample(formatted, target_size)

            all_data.extend(formatted)

    return Dataset.from_list(all_data)

def push_dataset(dataset, repo_id: str):
    print(f"\n🚀 Pushing dataset to: https://huggingface.co/datasets/{repo_id}")
    dataset.push_to_hub(repo_id)
    print("✅ Upload complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", type=str, default="balanced", choices=COMBOS.keys())
    parser.add_argument("--repo_id", type=str, default='koreankiwi99/mnlp_stem', help="HuggingFace dataset repo ID (e.g. username/sft-stem-v1)")
    args = parser.parse_args()

    print(f"🔧 Building '{args.combo}' SFT dataset ...")
    ds = build_combo(args.combo)
    print(f"📊 Final dataset size: {len(ds)} examples")

    push_dataset(ds, f'{args.repo_id}_{args.combo}')