import json
import argparse
import os

def qwen_format_prompt(prompt):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Original .jsonl file")
    args = parser.parse_args()

    # Ensure /processed directory exists
    os.makedirs("processed", exist_ok=True)

    # Get file name and create output path
    file_name = os.path.basename(args.input_path)
    output_path = os.path.join("processed", file_name)

    # Convert and save
    with open(args.input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            ex = json.loads(line)
            ex["prompt"] = qwen_format_prompt(ex["prompt"])
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Converted and saved to {output_path}")

if __name__ == "__main__":
    main()