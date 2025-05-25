import argparse
import os
import random
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_public", type=str, default="Vezora/Code-Preference-Pairs",
                        help="Name of a public Hugging Face dataset to use")
    parser.add_argument("--hf_username", type=str, default="koreankiwi99", help="Hugging Face username")
    parser.add_argument("--max_train_samples", type=int, default=2048,
                        help="Truncate the dataset to this number of samples.")
    args = parser.parse_args()

    # Load dataset
    print(f"📂 Loading dataset: {args.use_public}")
    raw_dataset = load_dataset(args.use_public, split="train")
    dataset_name = args.use_public.split("/")[-1]

    # Preprocessing for Code-Preference-Pairs
    def preprocess(example):
        return {
            "prompt": example["input"],
            "chosen": example["accepted"],
            "rejected": example["rejected"]
        }

    dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)
    dataset = dataset.shuffle(seed=42)

    # Truncate
    if args.max_train_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_train_samples)))
        print(f"📉 Truncated dataset to {len(dataset)} samples")

    # Add dataset name column
    dataset = dataset.add_column("dataset", [dataset_name] * len(dataset))

    dataset_size = len(dataset)
    base_model = "Qwen/Qwen3-0.6B-Base"

    # Repo names
    model_repo = f"{args.hf_username}/MNLP_M2_dpo_model"
    dataset_repo = f"{args.hf_username}/MNLP_M2_dpo_dataset"
    output_dir = f"./{model_repo.replace('/', '_')}"

    print(f"✅ Loaded {dataset_size} examples")
    print(f"🚀 Model will be pushed to: {model_repo}")
    print(f"🚀 Dataset will be pushed to: {dataset_repo}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = None

    # Load model and reference model
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

    # DPO config
    config = DPOConfig(
        beta=0.1,
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_length=512,
        max_prompt_length=128,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        output_dir=output_dir,
        remove_unused_columns=False,
        fp16=True,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
    )

    # Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Push model
    print("📤 Pushing model to 🤗 Hub...")
    model.push_to_hub(model_repo)
    tokenizer.push_to_hub(model_repo)
    print("✅ Model pushed!")

    # Push dataset
    print("📤 Pushing dataset to 🤗 Hub...")
    DatasetDict({"train": dataset}).push_to_hub(dataset_repo)
    print("✅ Dataset pushed!")


if __name__ == "__main__":
    main()