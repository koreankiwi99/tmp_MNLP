import random
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login

# Optional: log in if needed
# login(token="hf_...")

random_seed = 42

# Preprocess HuggingFaceH4/ultrafeedback_binarized
def preprocess_ultrafeedback(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"][1]["content"],
        "rejected": example["rejected"][1]["content"],
        "dataset": "ultrafeedback_binarized"
    }

# Preprocess Vezora/Code-Preference-Pairs
def preprocess_vezora(example):
    return {
        "prompt": example["input"],
        "chosen": example["accepted"],
        "rejected": example["rejected"],
        "dataset": "vezora_code_pairs"
    }

# Preprocess local dataset
def preprocess_local(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
        "dataset": "M1_dataset_v2"
    }

# Preprocess xinlai/Math-Step-DPO-10K
def preprocess_mathstep(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
        "dataset": "math_step_dpo_10K"
    }

# Load ultrafeedback
print("📥 Loading ultrafeedback_binarized...")
ultra = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
ultra = ultra.map(preprocess_ultrafeedback, remove_columns=ultra.column_names)

# Load Vezora
print("📥 Loading Vezora/Code-Preference-Pairs...")
vezora = load_dataset("Vezora/Code-Preference-Pairs", split="train")
vezora = vezora.map(preprocess_vezora, remove_columns=vezora.column_names)

# Load local .jsonl
print("📥 Loading your local dataset...")
local = load_dataset("json", data_files="./data/p_data_consistent.jsonl", split="train")
local = local.map(preprocess_local, remove_columns=local.column_names)

# Load Math-Step-DPO-10K
print("📥 Loading xinlai/Math-Step-DPO-10K...")
mathstep = load_dataset("xinlai/Math-Step-DPO-10K", split="train")
mathstep = mathstep.map(preprocess_mathstep, remove_columns=mathstep.column_names)

# Combine all
print("🔗 Combining datasets...")
combined = concatenate_datasets([ultra, vezora, local, mathstep])
combined = combined.shuffle(seed=random_seed)

# Summary
print(f"✅ Total examples: {len(combined)}")
print("🔍 Sample entry:", combined[0])

# Push to your repo
repo_name = "koreankiwi99/mnlp_aggregate_with_math"
print(f"📤 Uploading to: {repo_name}...")
combined.push_to_hub(repo_name, private=False)
print("✅ Upload complete!")