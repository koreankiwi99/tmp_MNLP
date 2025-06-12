from datasets import load_dataset
from collections import defaultdict, OrderedDict

# Hugging Face dataset paths for each variant
dataset_variants = {
    "lightweight": "koreankiwi99/mnlp_stem_lightweight",
    "balanced": "koreankiwi99/mnlp_stem_balanced",
    "math_only": "koreankiwi99/mnlp_stem_math_only",
    "reasoning": "koreankiwi99/mnlp_stem_reasoning",
    "curriculum": "koreankiwi99/mnlp_stem_curriculum"
}

# Mapping from raw dataset field values → LaTeX-friendly names
dataset_name_map = {
    "sciq": "SciQ",
    "openbookqa_main": "OpenBookQA",
    "mbpp": "MBPP",
    "mathinstruct": "MathInstruct",
    "hendrycks_math": "HendrycksMath"
}

# These are the column headers in the LaTeX table
all_sources = list(dataset_name_map.values())

# Count examples per dataset source for each variant
table = OrderedDict()

for variant, hf_path in dataset_variants.items():
    print(f"Loading {variant} from {hf_path}...")
    ds = load_dataset(hf_path, name="default", split="train")

    counts = defaultdict(int)

    for raw_name in ds["dataset"]:
        norm_key = raw_name.strip().lower()

        # Exact match
        if norm_key in dataset_name_map:
            pretty_name = dataset_name_map[norm_key]
            counts[pretty_name] += 1

        # Prefix match for HendrycksMath variants
        elif norm_key.startswith("hendrycks_math"):
            counts["HendrycksMath"] += 1

        else:
            print(f"[Warning] Unknown dataset source: {norm_key}")

    row = {source: counts.get(source, 0) for source in all_sources}
    row["Total"] = sum(row.values())
    table[variant] = row


# Format number for LaTeX: 0 → --, 10000 → 10K, 10500 → 10.5K
def fmt(n):
    if n == 0:
        return "--"
    elif n >= 1000:
        val = round(n / 1000, 1)
        return f"{val:.1f}K" if val % 1 else f"{int(val)}K"
    return str(n)

# Generate LaTeX table
latex = "\\begin{table}[h]\n\\centering\n\\resizebox{\\columnwidth}{!}{\n"
latex += "\\begin{tabular}{l|" + "c" * len(all_sources) + "c}\n"
latex += "\\toprule\n"
latex += " & ".join(["\\textbf{Variant}"] + [f"\\textbf{{{src}}}" for src in all_sources] + ["\\textbf{Total}"]) + " \\\\\n"
latex += "\\midrule\n"

for variant, row in table.items():
    line = " & ".join([f"\\texttt{{{variant}}}"] + [fmt(row[src]) for src in all_sources] + [fmt(row["Total"])])
    latex += line + " \\\\\n"

latex += "\\bottomrule\n\\end{tabular}}\n"
latex += "}\n\\caption{Source composition for each SFT variant (number of examples per domain).}\n"
latex += "\\label{tab:dpo_sft_variants}\n\\end{table}"

# Print LaTeX code
print("\n" + latex)
