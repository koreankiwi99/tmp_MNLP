from datasets import load_dataset, Dataset
import pandas as pd
from huggingface_hub import create_repo

def get_turn1_prompt_and_response(convo):
    """Extract the user message at turn 1 and assistant response at turn 1."""
    if len(convo) < 2:
        return None, None
    if convo[-2]["role"] != "user" or convo[-1]["role"] != "assistant":
        return None, None
    prompt = convo[-2]["content"].strip()
    response = convo[-1]["content"].strip()
    return prompt, response

def convert_mtbench_by_turn1_instruction(split="human"):
    ds = load_dataset("lmsys/mt_bench_human_judgments", split=split)
    records = []

    for ex in ds:
        if ex["winner"] not in ["model_a", "model_b"]:
            continue

        prompt_a, answer_a = get_turn1_prompt_and_response(ex["conversation_a"])
        prompt_b, answer_b = get_turn1_prompt_and_response(ex["conversation_b"])

        if not prompt_a or not prompt_b or prompt_a != prompt_b:
            continue  # skip mismatched instruction

        chosen = answer_a if ex["winner"] == "model_a" else answer_b
        rejected = answer_b if ex["winner"] == "model_a" else answer_a

        records.append({
            "prompt": prompt_a,
            "chosen": chosen,
            "rejected": rejected
        })

    return Dataset.from_pandas(pd.DataFrame(records))

# Save or push
for split in ["human", "gpt4_pair"]:
    dpo_dataset = convert_mtbench_by_turn1_instruction(split)
    HF_REPO_ID = f"koreankiwi99/mtbench-dpo-turn1-{split}"
    create_repo(HF_REPO_ID, repo_type="dataset", exist_ok=True)
    dpo_dataset.push_to_hub(HF_REPO_ID)