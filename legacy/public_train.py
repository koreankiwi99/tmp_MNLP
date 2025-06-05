import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_username", type=str, default="koreankiwi99", help="Hugging Face username")
    args = parser.parse_args()

    dataset_name = "webgpt"
    model_repo = f"{args.hf_username}/dpo_model_{dataset_name}"
    output_dir = f"./{model_repo.replace('/', '_')}"

    print(f"📘 Dataset: {dataset_name}")
    print(f"🚀 Model will be pushed to: {model_repo}")

    # Load and preprocess WebGPT
    raw_dataset = load_dataset("openai/webgpt_comparisons", split="train")

    def reformat_webgpt(example):
        prompt = example["question"]["full_text"]
        a0 = example["answer_0"]
        a1 = example["answer_1"]
        if example["score_0"] > example["score_1"]:
            return {"prompt": prompt, "chosen": a0, "rejected": a1}
        else:
            return {"prompt": prompt, "chosen": a1, "rejected": a0}

    dataset = raw_dataset.map(reformat_webgpt)
    print(f"✅ Loaded and reformatted {len(dataset)} examples")
    print(dataset[0])

    # Tokenizer
    base_model = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Models
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        #torch_dtype=torch.float16,
        device_map="auto"
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        #torch_dtype=torch.float16,
        device_map="auto"
    )

    # DPO Config
    config = DPOConfig(
        beta=0.1,
        learning_rate=5e-7,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_length=512,
        max_prompt_length=128,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        output_dir=output_dir,
        remove_unused_columns=True,  # ✅ important
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
    )

    # Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,  # ✅ tokenizer used for internal processing
    )

    # Train
    trainer.train()

    # Push to hub
    print("📤 Pushing model to 🤗 Hub...")
    model.push_to_hub(model_repo)
    tokenizer.push_to_hub(model_repo)
    print("✅ Done!")

if __name__ == "__main__":
    main()
