import json
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "memorization_score", "results")

DEEPSEEK_PATH = os.path.join(RESULTS_DIR, "deepseek_llm_7b_perplexity.json")
LLAMA_PATH = os.path.join(RESULTS_DIR, "llama_2_7b_hf_perplexity.json")
MISTRAL_PATH = os.path.join(RESULTS_DIR, "mistral-7b_v01_perplexity.json")

MODEL_PATHS = {
    "DeepSeek LLM 7B": DEEPSEEK_PATH,
    "Llama 7B": LLAMA_PATH,
    "Mistral 7B": MISTRAL_PATH
}

OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots", "buckets_4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def bucket_label(x):
    if x < 10:
        return "good"
    elif x < 29.5:
        return "average"
    elif x < 100:
        return "poor"

    else:
        return "nan"

def plot_buckets(indices, buckets, model_name):
    colors = {"good": "#6abd63", "average": "#ffe156", "poor": "#ff595e"}
    bucket_names = ["good", "average", "poor"]

    plt.figure(figsize=(10, 6))

    for bucket in bucket_names:
        bucket_data = [v for v, b in zip(indices, buckets) if b == bucket]
        plt.hist(
            bucket_data,
            bins=50,
            alpha=0.8,
            label=f"{bucket} ({len(bucket_data)})",
            color=colors[bucket],
            edgecolor='black'
        )

    plt.title(f"Histogram of Adjusted Perplexity by Bucket ({model_name})")
    plt.xlabel("Adjusted Perplexity")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name.lower().replace(' ', '_')}_adjusted_perplexity_buckets_hist.png"
    )
    plt.savefig(out_path)
    plt.close()

def get_adjusted_perplexities_and_buckets(data):
    indices = []
    for item in data["results"]:
        results = item.get("results", {})
        adj_perp = results.get("adjusted_perplexity")
        if adj_perp is not None:
            indices.append(adj_perp)
    buckets = [bucket_label(x) for x in indices]
    mean = np.mean(indices)
    std = np.std(indices)
    return indices, buckets, mean, std

def process_and_plot(json_path, model_name):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    indices, buckets, mean, std = get_adjusted_perplexities_and_buckets(data)
    print(f"\n{model_name}:")
    print(f"Mean adjusted_perplexity: {mean:.4f}")
    print(f"Std adjusted_perplexity: {std:.4f}")
    print(f"Good: {buckets.count('good')}")
    print(f"Average: {buckets.count('average')}")
    print(f"Poor: {buckets.count('poor')}")
    plot_buckets(indices, buckets, model_name)

if __name__ == "__main__":
    for model_name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            process_and_plot(path, model_name)
        else:
            print(f"Warning: File not found for {model_name} at {path}")

