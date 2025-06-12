from datasets import load_dataset

# Load datasets
ds1 = load_dataset("koreankiwi99/MNLP_M3_dpo_dataset")
ds2 = load_dataset("zechen-nlp/MNLP_dpo_demo")#"zechen-nlp/MNLP_dpo_evals")

# Pick split (adjust if your datasets use other split names)
prompts1 = set(ds1["train"]['prompt'])
prompts2 = set(ds2["test"]['prompt'])

# Find overlaps
overlap = prompts1.intersection(prompts2)

print(f"Number of overlapping prompts: {len(overlap)}")
if overlap:
    print("Sample overlapping prompt(s):")
    for i, prompt in enumerate(list(overlap)[:5]):  # Show up to 5 examples
        print(f"{i+1}. {prompt}")
else:
    print("No overlapping prompts found.")