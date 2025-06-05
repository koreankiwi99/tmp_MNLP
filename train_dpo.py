import argparse
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from huggingface_hub import HfApi, Repository


def preprocess(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to DPO config JSON file")
    parser.add_argument("--hf_dataset_name", type=str, required=True, help="Hugging Face dataset name (e.g. HuggingFaceH4/ultrafeedback_binarized)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B-Base", help="Base model name or path")
    parser.add_argument("--hf_username", type=str, default="koreankiwi99", help="Hugging Face username")
    args = parser.parse_args()

    # Load and preprocess dataset
    print(f"📂 Loading dataset: {args.hf_dataset_name}")
    dataset_name = args.hf_dataset_name.split('/')[-1]
    dataset = load_dataset(args.hf_dataset_name, split="train")
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names).shuffle(seed=42)

    # Load config
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    # Set output directory and model repo
    config_name = args.config_path.split('/')[-1].replace('.json', '')
    model_repo = f"{args.hf_username}/dpo_model_{config_name}_{dataset_name}"
    output_dir = f"./{model_repo.replace('/', '_')}"
    config_dict["output_dir"] = output_dir
    config_dict["push_to_hub_model_id"] = model_repo

    config = DPOConfig(**config_dict)

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = None

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    # Train the model
    trainer.train()

    # Save and push model, tokenizer, and config
    print("📤 Pushing model and config to 🤗 Hub...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "dpo_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Initialize and push to the hub (avoiding clone_from error)
    repo = Repository(local_dir=output_dir)
    repo.git_init()
    repo.create_remote(repo_url=f"https://huggingface.co/{model_repo}")
    repo.push_to_hub()

if __name__ == "__main__":
    main()