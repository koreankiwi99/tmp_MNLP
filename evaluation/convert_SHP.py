from datasets import load_dataset, Dataset
from huggingface_hub import create_repo
import json

# Replace with your Hugging Face username and desired repository name
HF_USERNAME = "koreankiwi99"
REPO_NAME = "shp-dpo-eval"
FULL_REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# Load the SHP dataset (choose 'train', 'validation', or 'test' split)
shp_dataset = load_dataset("stanfordnlp/SHP", split="test")

# Convert to desired format
converted_samples = []
for example in shp_dataset:
    prompt = example["history"]
    if example["labels"] == 1:
        chosen = example["human_ref_A"]
        rejected = example["human_ref_B"]
    elif example["labels"] == 0:
        chosen = example["human_ref_B"]
        rejected = example["human_ref_A"]
    else:
        continue  # Skip if label is invalid
    converted_samples.append({
        "prompt": prompt.strip(),
        "chosen": chosen.strip(),
        "rejected": rejected.strip()
    })

# Create Hugging Face-compatible Dataset object
converted_dataset = Dataset.from_list(converted_samples)

# Create repository on Hugging Face Hub (if it doesn't exist)
create_repo(FULL_REPO_ID, repo_type="dataset", exist_ok=True)

# Push the dataset to the Hugging Face Hub
converted_dataset.push_to_hub(FULL_REPO_ID)

print(f"✅ Uploaded to: https://huggingface.co/datasets/{FULL_REPO_ID}")