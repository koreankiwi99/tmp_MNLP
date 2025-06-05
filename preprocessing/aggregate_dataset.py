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

# Preprocess your local dataset
def preprocess_local(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
        "dataset": example["dataset"]
    }

# Load ultrafeedback
print("📥 Loading ultrafeedback_binarized...")
ultra = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
ultra = ultra.map(preprocess_ultrafeedback, remove_columns=ultra.column_names)

# Load Vezora
print("📥 Loading Vezora/Code-Preference-Pairs...")
vezora = load_dataset("Vezora/Code-Preference-Pairs", split="train")
vezora = vezora.map(preprocess_vezora, remove_columns=vezora.column_names)

# Load your local .jsonl
print("📥 Loading your local dataset...")
local = load_dataset("json", data_files="./data/p_data_consistent.jsonl", split="train")
local = local.map(preprocess_local, remove_columns=local.column_names)

# Combine all
print("🔗 Combining datasets...")
combined = concatenate_datasets([ultra, vezora, local])
combined = combined.shuffle(seed=random_seed)

# Summary
print(f"✅ Total examples: {len(combined)}")
print("🔍 Sample entry:", combined[0])

# Push to your repo
repo_name = "koreankiwi99/mnlp_aggregate"
print(f"📤 Uploading to: {repo_name}...")
combined.push_to_hub(repo_name, private=False)
print("✅ Upload complete!")