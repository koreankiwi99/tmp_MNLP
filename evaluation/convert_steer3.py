from datasets import load_dataset, Dataset
import pandas as pd
from huggingface_hub import create_repo

def extract_user_prompt(context):
    """Concatenate only the user messages in the context as the prompt."""
    return "\n".join([msg["content"].strip() for msg in context if msg["role"] == "user"]).strip()

def convert_helpsteer3(domain_filter):
    """Convert HelpSteer3 to DPO format with clean prompt (no roles)."""
    ds = load_dataset("nvidia/HelpSteer3", split="validation", name="preference")
    records = []

    for ex in ds:
        score = ex["overall_preference"]
        if score == 0 or ex["domain"] != domain_filter:
            continue

        prompt = extract_user_prompt(ex["context"])
        if not prompt:
            continue

        r1, r2 = ex["response1"].strip(), ex["response2"].strip()
        chosen, rejected = (r2, r1) if score > 0 else (r1, r2)

        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    return Dataset.from_pandas(pd.DataFrame(records))

# Push to Hugging Face Hub: STEM and CODE domains
for domain in ["code", "stem", "general"]:
    dataset = convert_helpsteer3(domain)
    HF_REPO_ID = f"koreankiwi99/helpsteer3-dpo-{domain}"
    create_repo(HF_REPO_ID, repo_type="dataset", exist_ok=True)
    dataset.push_to_hub(HF_REPO_ID)
    print(f"✅ Uploaded {len(dataset)} examples → {HF_REPO_ID}")