import os
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "memorization_score", "results")

DATASET_PATHS = {
    "deepseek_llm_7b": os.path.join(DATASET_DIR, "deepseek_llm_7b_perplexity.json"),
    "llama_2_7b_hf": os.path.join(DATASET_DIR, "llama_2_7b_hf_perplexity.json"),
    "mistral_7b_v01": os.path.join(DATASET_DIR, "mistral-7b_v01_perplexity.json"),
}

OUTPUT_DIR = os.path.join(DATASET_DIR, "histograms_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_dataset(model_name: str, file_path: str):
    print(f"Processing {model_name} from {file_path} ...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    perplexities = [item["results"]["adjusted_perplexity"] for item in data["results"]]
    token_counts = [item["results"]["token_count"] for item in data["results"]]

    # Histogram perplexity
    plt.figure()
    plt.hist(perplexities, bins=30, color="skyblue", edgecolor="black")
    plt.title(f"{model_name} - Perplexity Distribution")
    plt.xlabel("Perplexity")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_perplexity_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Histogram token count
    plt.figure()
    plt.hist(token_counts, bins=30, color="salmon", edgecolor="black")
    plt.title(f"{model_name} - Token Count Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_token_count_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # # 2D histogram perplexity vs length
    # plt.figure()
    # plt.hist2d(token_counts, perplexities, bins=(30, 30), cmap="viridis")
    # plt.colorbar(label="Number of Samples")
    # plt.title(f"{model_name} - Perplexity Distribution over Text Length")
    # plt.xlabel("Token Count (Text Length)")
    # plt.ylabel("Perplexity")
    # plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_perplexity_vs_length_hist2d.png"), dpi=150, bbox_inches="tight")
    # plt.close()

    # ⬇⬇ Nowy wykres: Perplexity vs Token Count (średnia w przedziałach)
    bins = np.linspace(min(token_counts), max(token_counts), 10)  # 30 przedziałów długości
    bin_indices = np.digitize(token_counts, bins)
    bin_means = []
    bin_centers = []

    for i in range(1, len(bins)):
        bin_perplexities = [p for p, b in zip(perplexities, bin_indices) if b == i]
        if bin_perplexities:
            bin_means.append(np.mean(bin_perplexities))
            # środek przedziału do osi X
            bin_centers.append((bins[i-1] + bins[i]) / 2)

    # plt.figure()
    # plt.plot(bin_centers, bin_means, marker="o", linestyle="-", color="green")
    # plt.title(f"{model_name} - Average Perplexity vs Text Length")
    # plt.xlabel("Token Count (binned)")
    # plt.ylabel("Average Perplexity")
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_perplexity_vs_length_line.png"), dpi=150, bbox_inches="tight")
    # plt.close()

    # print(f"Saved histograms and line plot for {model_name} in {OUTPUT_DIR}\n")


    #boxplot

    fig, axs = plt.subplots(1, len(bins), figsize=(30, 5))
    for i in range(1, len(bins)):
        bin_perplexities = [p for p, b in zip(perplexities, bin_indices) if b == i]

        axs[i].boxplot(np.array(bin_perplexities))
        axs[i].set_title("")
        axs[i].set_xlabel(f"{bin_indices}")
        axs[i].set_ylabel("Val")

    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_boxplots.png"))



for model_name, path in DATASET_PATHS.items():
    if os.path.exists(path):
        process_dataset(model_name, path)
    else:
        print(f"File not found: {path}")
