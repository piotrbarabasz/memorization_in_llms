import os
import json
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "memorization_score", "results")

DATASET_PATHS = {
    "deepseek_llm_7b": os.path.join(DATASET_DIR, "deepseek_llm_7b_perplexity.json"),
    "llama_2_7b_hf": os.path.join(DATASET_DIR, "llama_2_7b_hf_perplexity.json"),
    "mistral_7b_v01": os.path.join(DATASET_DIR, "mistral-7b_v01_perplexity.json"),
}

OUTPUT_DIR = os.path.join(DATASET_DIR, "histograms")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_dataset(model_name: str, file_path: str):
    print(f"Processing {model_name} from {file_path} ...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    perplexities = [item["results"]["perplexity"] for item in data["results"]]
    token_counts = [item["results"]["token_count"] for item in data["results"]]

    # Perplexity histogram
    plt.figure()
    plt.hist(perplexities, bins=30, color="skyblue", edgecolor="black")
    plt.title(f"{model_name} - Perplexity Distribution")
    plt.xlabel("Perplexity")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_perplexity_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Token count histogram
    plt.figure()
    plt.hist(token_counts, bins=30, color="salmon", edgecolor="black")
    plt.title(f"{model_name} - Token Count Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_token_count_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Perplexity vs Text Length (2D histogram)
    plt.figure()
    plt.hist2d(token_counts, perplexities, bins=(30, 30), cmap="viridis")
    plt.colorbar(label="Number of Samples")
    plt.title(f"{model_name} - Perplexity Distribution over Text Length")
    plt.xlabel("Token Count (Text Length)")
    plt.ylabel("Perplexity")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_perplexity_vs_length_hist2d.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved histograms for {model_name} in {OUTPUT_DIR}\n")


for model_name, path in DATASET_PATHS.items():
    if os.path.exists(path):
        process_dataset(model_name, path)
    else:
        print(f"File not found: {path}")

# import os
# import json
# import matplotlib.pyplot as plt

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATASET_DIR = os.path.join(BASE_DIR, "memorization_score", "results")

# DEEPSEEK_PATH = os.path.join(DATASET_DIR, "deepseek_llm_7b_perplexity.json")
# LLAMA_PATH = os.path.join(DATASET_DIR, "llama_2_7b_hf_perplexity.json")
# MISTRAL_PATH = os.path.join(DATASET_DIR, "mistral-7b_v01_perplexity.json")

# OUTPUT_FILE =  os.path.join(DATASET_DIR, "histograms")

# with open(DATASET_PATH, "r", encoding="utf-8") as f:
#     data = json.load(f)

# perplexities = [item["results"]["perplexity"] for item in data["results"]]
# token_counts = [item["results"]["token_count"] for item in data["results"]]

# plt.hist(perplexities, bins=30, color="skyblue", edgecolor="black")
# plt.title("Perplexity Distribution")
# plt.xlabel("Perplexity")
# plt.ylabel("Frequency")
# plt.savefig(os.path.join(OUTPUT_FILE, "perplexity_hist.png"), dpi=150, bbox_inches="tight")
# plt.close()

# plt.hist(token_counts, bins=30, color="salmon", edgecolor="black")
# plt.title("Token Count Distribution")
# plt.xlabel("Token Count")
# plt.ylabel("Frequency")
# plt.savefig(os.path.join(OUTPUT_FILE, "token_count_hist.png"), dpi=150, bbox_inches="tight")
# plt.close()