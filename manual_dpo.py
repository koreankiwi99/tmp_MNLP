import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to Qwen-formatted DPO dataset (.jsonl)")
    parser.add_argument("--hf_username", type=str, default="koreankiwi99", help="Hugging Face username")
    args = parser.parse_args()

    # Derived names
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    model_repo = f"{args.hf_username}/dpo_model_{dataset_name}"
    output_dir = f"./{model_repo.replace('/', '_')}"

    print(f"📘 Dataset: {dataset_name}")
    print(f"🚀 Model will be pushed to: {model_repo}")

    # Load dataset
    raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
    print(f"✅ Loaded {len(raw_dataset)} examples")

    # Load tokenizer
    base_model = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token  # Needed for batching

    # Manual tokenization (avoid chat template)
    def tokenize(example):
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        prompt_ids = tokenizer(prompt, truncation=True, max_length=128, add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(chosen, truncation=True, max_length=384, add_special_tokens=False)["input_ids"]
        rejected_ids = tokenizer(rejected, truncation=True, max_length=384, add_special_tokens=False)["input_ids"]

        return {
            "prompt_input_ids": prompt_ids,
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids,
        }

    dataset = raw_dataset.map(tokenize, remove_columns=raw_dataset.column_names)

    # Debug a few entries
    print("🔍 Sample tokenized example:")
    print(dataset[0])

    # Load model and reference model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # DPO training config
    config = DPOConfig(
        beta=0.1,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_length=512,
        max_prompt_length=128,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        output_dir=output_dir,
        remove_unused_columns=False,
        fp16=False,  # Use False unless you're confident fp16 gradients work
        gradient_checkpointing=True,
        is_dataset_tokenized=True,
    )

    # DPOTrainer (no chat template!)
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
    )

    # Enable debugging for gradient anomalies
    torch.autograd.set_detect_anomaly(True)

    # Train
    trainer.train()

    # Push model and tokenizer to Hugging Face Hub
    print("📤 Pushing model to 🤗 Hub...")
    model.push_to_hub(model_repo)
    tokenizer.push_to_hub(model_repo)
    print("✅ Done!")


if __name__ == "__main__":
    main()