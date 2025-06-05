from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from tqdm import tqdm


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
            ans = choices[ord(ex["answerKey"]) - ord("A")]
            formatted.append({
                "instruction": f"{q} Choices: {', '.join(choices)}",
                "output": ans,
                "dataset": dataset_name
            })

    elif dataset_name == "hendrycks_math_algebra":
        for ex in dataset:
            formatted.append({
                "instruction": ex["problem"],
                "output": ex["solution"],
                "dataset": dataset_name
            })

    elif dataset_name == "mbpp":
        for ex in dataset:
            formatted.append({
                "instruction": ex["text"],
                "output": ex["code"],
                "dataset": dataset_name
            })

    elif dataset_name == "code_contests":
        for ex in dataset:
            formatted.append({
                "instruction": ex["prompt"],
                "output": ex["canonical_solution"],
                "dataset": dataset_name
            })

    elif dataset_name == "proofwriter_depth_3":
        for ex in dataset:
            formatted.append({
                "instruction": ex["question"],
                "output": ex["proof"],
                "dataset": dataset_name
            })

    elif dataset_name == "pubmedqa":
        for ex in dataset:
            if ex["final_decision"] in ["yes", "no", "maybe"]:
                formatted.append({
                    "instruction": f"{ex['question']}\n\n{ex['context']}",
                    "output": ex["final_decision"],
                    "dataset": dataset_name
                })

    elif dataset_name == "MathInstruct":
        for ex in dataset:
            formatted.append({
                "instruction": ex["instruction"],
                "output": ex["output"],
                "dataset": dataset_name
            })

    elif dataset_name == "OpenMathInstruct-1":
        for ex in dataset:
            formatted.append({
                "instruction": ex["question"],
                "output": ex["answer"],
                "dataset": dataset_name
            })

    return Dataset.from_list(formatted)


def build_combo(combo_id):
    sources = []

    if combo_id == "1":  # balanced
        sources = [
            ("sciq", "train"),
            ("openbookqa", "main", "train"),
            ("mbpp", "train"),
            ("deepmind/code_contests", "train"),
            ("proofwriter", "depth_3", "train"),
            ("hendrycks_math", "algebra", "train"),
            ("TIGER-Lab", "MathInstruct", "train"),
            ("nvidia", "OpenMathInstruct-1", "train")
        ]

    elif combo_id == "2":  # code-focused
        sources = [
            ("mbpp", "train"),
            ("deepmind/code_contests", "train")
        ]

    elif combo_id == "3":  # logic + math
        sources = [
            ("sciq", "train"),
            ("proofwriter", "depth_3", "train"),
            ("hendrycks_math", "algebra", "train"),
            ("TIGER-Lab", "MathInstruct", "train")
        ]

    elif combo_id == "4":  # lightweight mix
        sources = [
            ("sciq", "train"),
            ("mbpp", "train")
        ]

    datasets = []
    for s in sources:
        if len(s) == 2:
            name, split = s
            ds = load_dataset(name, split=split)
            ds_name = name.split("/")[-1]
        else:
            name, subset, split = s
            ds = load_dataset(name, subset, split=split)
            ds_name = subset
        datasets.append(format_dataset_dpo_style(ds, ds_name))

    return Dataset.concatenate(*datasets)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", type=str, default="1")
    parser.add_argument("--repo_id", type=str, default="koreankiwi99/mnlp_sft_combo")
    args = parser.parse_args()

    repo_id = f'{args.repo_id}_{args.combo}'
    print(f"🚀 Building combo {args.combo} ...")
    final_ds = build_combo(args.combo)
    final_ds.push_to_hub(repo_id)
    print(f"✅ Pushed to HuggingFace Hub → https://huggingface.co/datasets/{repo_id}")