"""
Create histograms based on fixed lowerbound ~1014 tokens
"""

import json
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# DATASET_PATH = os.path.join(BASE_DIR, "dataset", "extracted_wikipedia_token_limit_records_5000.json")
MODEL_FILES = {
    "deepseek_llm_7b": os.path.join(RESULTS_DIR, "deepseek_llm_7b_perplexity.json"),
    "llama_2_7b_hf": os.path.join(RESULTS_DIR, "llama_2_7b_hf_perplexity.json"),
    "mistral-7b_v01": os.path.join(RESULTS_DIR, "mistral-7b_v01_perplexity.json"),
}

# BUCKETS_DIR = os.path.join(RESULTS_DIR, "buckets_100_samples_avg_poor")
# os.makedirs(BUCKETS_DIR, exist_ok=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PURE_HIST_DIR = os.path.join(SCRIPT_DIR, "pure_histograms_token_lower_bound_1000")
os.makedirs(PURE_HIST_DIR, exist_ok=True)

def load_adjusted_perplexities(path: str):
    """Return list of adjusted_perplexity values from a result JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    values = []
    for item in data["results"]:
        inner = item.get("results", {})

        token_count = inner.get("token_count", 0)
        if token_count <= 1000:
            continue
        
        if "adjusted_perplexity" in inner:
            values.append(inner["adjusted_perplexity"])
        else:
            # optional: debug if some record is missing the field
            print(f"[WARN] Missing 'adjusted_perplexity' in item id={item.get('id')}")
    return values


def save_histogram(values, model_name: str):
    """Create and save histogram for a given model."""
    plt.figure()
    plt.hist(values, bins=30)  # you can tweak number of bins if needed
    plt.title(f"Adjusted Perplexity Histogram - {model_name}")
    plt.xlabel("Adjusted Perplexity")
    plt.ylabel("Frequency")

    out_path = os.path.join(PURE_HIST_DIR, f"{model_name}_adjusted_perplexity_hist.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved histogram for {model_name} → {out_path}")


def main():
    for model_name, json_path in MODEL_FILES.items():
        if not os.path.exists(json_path):
            print(f"[SKIP] File not found for {model_name}: {json_path}")
            continue

        values = load_adjusted_perplexities(json_path)
        if not values:
            print(f"[WARN] No adjusted_perplexity values for {model_name}")
            continue

        save_histogram(values, model_name)


if __name__ == "__main__":
    main()