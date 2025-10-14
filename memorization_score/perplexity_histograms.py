import json
import os
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

OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots", "histograms")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_and_save_histogram(json_path, model_name):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    adjusted_perplexities = [
        item["results"]["adjusted_perplexity"]
        for item in data["results"]
        if "adjusted_perplexity" in item["results"]
    ]
    plt.figure(figsize=(10, 6))
    plt.hist(adjusted_perplexities, bins=50, edgecolor='black')
    plt.title(f"Histogram of Adjusted Perplexity ({model_name})")
    plt.xlabel("Adjusted Perplexity")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_adjusted_perplexity_hist.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot for {model_name} to {out_path}")

for model_name, path in MODEL_PATHS.items():
    plot_and_save_histogram(path, model_name)
