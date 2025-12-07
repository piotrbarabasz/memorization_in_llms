import json
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Paths (your existing setup) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_FILES = {
    "deepseek_llm_7b": os.path.join(RESULTS_DIR, "deepseek_llm_7b_perplexity.json"),
    "llama_2_7b_hf": os.path.join(RESULTS_DIR, "llama_2_7b_hf_perplexity.json"),
    "mistral-7b_v01": os.path.join(RESULTS_DIR, "mistral-7b_v01_perplexity.json"),
}

# Pretty names for plot titles
MODEL_LABELS = {
    "deepseek_llm_7b": "DeepSeek 7B",
    "llama_2_7b_hf": "Llama 7B",
    "mistral-7b_v01": "Mistral 7B",
}

# Bucket configs: (lower_bound, upper_bound_or_None, bucket_name)
BUCKET_CONFIG = {
    "deepseek_llm_7b": [
        (0, 10, "good"),
        (10, 30, "average"),
        (30, None, "poor"),
    ],
    "llama_2_7b_hf": [
        (0, 10, "good"),
        (10, 20, "average"),
        (20, None, "poor"),
    ],
    "mistral-7b_v01": [
        (0, 10, "good"),
        (10, 20, "average"),
        (20, None, "poor"),
    ],
}

# --- Output folder: in same location as this .py file ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUCKETED_HIST_DIR = os.path.join(SCRIPT_DIR, "bucketed_histograms")
os.makedirs(BUCKETED_HIST_DIR, exist_ok=True)


def load_adjusted_perplexities(path: str):
    """Return list of adjusted_perplexity values from a result JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    values = []
    for item in data["results"]:
        inner = item.get("results", {})
        if "adjusted_perplexity" in inner:
            values.append(inner["adjusted_perplexity"])
    return values


BUCKET_QUANTILES = {
    # good ≈ 0–78%, average ≈ 78–93%, poor ≈ 7% top tail
    "deepseek_llm_7b": (0.78, 0.93),
    "llama_2_7b_hf": (0.80, 0.94),
    "mistral-7b_v01": (0.80, 0.94),
}

USE_LOG_FOR_BUCKETS = True  # log-transform for heavy right-tailed dists


def _dynamic_bucket_cfg(model_name: str, values):
    """
    Build (lower, upper, name) tuples using quantiles instead of fixed numbers.
    """
    if model_name not in BUCKET_QUANTILES:
        # fallback to your hand-tuned config if no quantiles defined
        return BUCKET_CONFIG[model_name]

    good_q, avg_q = BUCKET_QUANTILES[model_name]

    arr = np.asarray(values, dtype=float)
    work = np.log(arr) if USE_LOG_FOR_BUCKETS else arr

    q_good = np.quantile(work, good_q)
    q_avg = np.quantile(work, avg_q)

    b1 = float(np.exp(q_good) if USE_LOG_FOR_BUCKETS else q_good)
    b2 = float(np.exp(q_avg)  if USE_LOG_FOR_BUCKETS else q_avg)

    # standard 3 buckets: [0, b1), [b1, b2), [b2, +inf)
    return [
        (0.0, b1, "good"),
        (b1, b2, "average"),
        (b2, None, "poor"),
    ]


def bucket_values(model_name: str, values):
    """
    Split values into good/average/poor using either dynamic quantile-based
    thresholds (preferred) or the fixed BUCKET_CONFIG as fallback.
    """
    cfg = _dynamic_bucket_cfg(model_name, values)
    buckets = {name: [] for (_, _, name) in cfg}

    for v in values:
        for lower, upper, name in cfg:
            if upper is None:
                if v >= lower:
                    buckets[name].append(v)
                    break
            else:
                if lower <= v < upper:
                    buckets[name].append(v)
                    break

    return buckets


def save_bucketed_histogram(model_name: str, values):
    if not values:
        print(f"[WARN] No values for {model_name}, skipping plot.")
        return

    buckets = bucket_values(model_name, values)

    # Common bins for all buckets so they line up nicely
    vmin, vmax = min(values), max(values)
    bins = np.linspace(vmin, vmax, 60)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Overlaid histograms for each bucket
    for bucket_name in ["good", "average", "poor"]:
        bucket_vals = buckets[bucket_name]
        if not bucket_vals:
            continue
        label = f"{bucket_name} ({len(bucket_vals)})"
        ax.hist(bucket_vals, bins=bins, alpha=0.7, label=label)

    ax.set_title(f"Histogram of Adjusted Perplexity by Bucket ({MODEL_LABELS[model_name]})")
    ax.set_xlabel("Adjusted Perplexity")
    ax.set_ylabel("Count")
    ax.legend()

    out_path = os.path.join(BUCKETED_HIST_DIR, f"{model_name}_bucketed_hist.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[OK] Saved bucketed histogram for {model_name} → {out_path}")


def main():
    for model_name, json_path in MODEL_FILES.items():
        if not os.path.exists(json_path):
            print(f"[SKIP] File not found for {model_name}: {json_path}")
            continue

        values = load_adjusted_perplexities(json_path)
        save_bucketed_histogram(model_name, values)


if __name__ == "__main__":
    main()
