from datasets import load_dataset, Dataset
from huggingface_hub import create_repo
import re

# Load Anthropic HH dataset
ds = load_dataset("Anthropic/hh-rlhf", split="test")  # or 'validation'

# Define regex to extract prompt and last assistant line
def split_prompt_and_last(text):
    """
    Splits into (prompt, last_assistant_reply)
    """
    # Captures everything up to the final Assistant reply
    match = re.match(r"^(.*Human:.*\n)Assistant:\s*(.*)$", text.rstrip(), re.DOTALL)
    if match:
        prompt = match.group(1).strip()
        last = match.group(2).strip()
        return prompt, last
    return None, None

# Filter and Convert
converted = []
for ex in ds:
    p_c, last_c = split_prompt_and_last(ex["chosen"])
    p_r, last_r = split_prompt_and_last(ex["rejected"])
    if p_c and p_c == p_r:
        converted.append({
            "prompt": p_c,
            "chosen": last_c,
            "rejected": last_r
        })


converted_ds = Dataset.from_list(converted)
print(f"🚀 Filtered down to {len(converted)} samples (prompt match)")


HF_USERNAME = "koreankiwi99"      
REPO_NAME = "hh-dpo-eval"         
FULL_ID = f"{HF_USERNAME}/{REPO_NAME}"

create_repo(FULL_ID, repo_type="dataset", exist_ok=True)
converted_ds.push_to_hub(FULL_ID)

print(f"✅ Uploaded to https://huggingface.co/datasets/{FULL_ID}")