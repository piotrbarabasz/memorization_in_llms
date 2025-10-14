import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "memorization_score", "results")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "extracted_wikipedia_token_limit_records_5000.json")
MODEL_FILES = {
    "deepseek_llm_7b": os.path.join(RESULTS_DIR, "deepseek_llm_7b_perplexity.json"),
    "llama_2_7b_hf": os.path.join(RESULTS_DIR, "llama_2_7b_hf_perplexity.json"),
    "mistral-7b_v01": os.path.join(RESULTS_DIR, "mistral-7b_v01_perplexity.json"),
}

BUCKETS_DIR = os.path.join(RESULTS_DIR, "buckets_100_samples_avg_poor")
os.makedirs(BUCKETS_DIR, exist_ok=True)

def bucket_label(adj_perplexity):
    """Return bucket name from adjusted_perplexity."""
    if adj_perplexity < 10:
        return "good"
    if adj_perplexity < 29.5:
        return "average"
    return "poor"


def load_wikipedia(path):
    """Return a mapping id → record for the Wikipedia slice."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {str(rec["id"]): rec for rec in records}


def select_closest(ids, results_by_id, target, k=100):
    """Return up to k ids whose token_count is closest to target."""
    ids_sorted = sorted(ids,
                        key=lambda _id: abs(results_by_id[_id]["token_count"] - target))
    return ids_sorted[:k]


def save_bucket_json(model, bucket, ids, avg_poor_tc, wiki_by_id, results_by_id):
    out = {
        "metadata": {
            "model": model,
            "bucket_bin": bucket,
            "dataset_size": len(ids),
            "avg_token_count_bucket_poor": avg_poor_tc,
        },
        "results": [],
    }

    for _id in ids:
        wiki  = wiki_by_id.get(_id, {})
        stats = results_by_id[_id]
        out["results"].append({
            "id": _id,
            "adjusted_perplexity": stats.get("adjusted_perplexity"),
            "token_count": stats.get("token_count"),
            "rare_token_ratio": stats.get("rare_token_ratio"),
            "category": wiki.get("category", ""),
            "title": wiki.get("title", ""),
            "text": wiki.get("text", ""),
        })

    dest_dir  = os.path.join(BUCKETS_DIR, model)
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, f"{model}_bucket_{bucket}.json")

    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"saved {len(ids):>3} → {os.path.relpath(dest_file, BASE_DIR)}")


def main():
    # Load base wiki slice
    wiki_by_id = load_wikipedia(DATASET_PATH)

    # Process each model
    for model, res_path in MODEL_FILES.items():
        print(f"\n▶ Processing {model}")

        if not os.path.exists(res_path):
            print(f"  ! missing: {res_path}")
            continue

        with open(res_path, "r", encoding="utf-8") as f:
            model_json = json.load(f)

        # id -> stats
        results_by_id = {str(item["id"]): item["results"]
                         for item in model_json["results"]}

        # raw buckets
        buckets = {"good": [], "average": [], "poor": []}
        for _id, stats in results_by_id.items():
            adj_ppl = stats.get("adjusted_perplexity")
            if adj_ppl is not None:
                buckets[bucket_label(adj_ppl)].append(_id)

        # compute poor-bucket average token_count
        poor_ids = buckets["poor"]
        avg_poor_tc = (sum(results_by_id[_id]["token_count"] for _id in poor_ids) /
                       len(poor_ids)) if poor_ids else 0.0

        # for each bucket keep 100 closest
        for bucket in ("good", "average", "poor"):
            ids = buckets[bucket]
            if not ids:
                print(f" no samples in {bucket} - skipped")
                continue

            selected = select_closest(ids, results_by_id, avg_poor_tc, k=100)
            save_bucket_json(model, bucket, selected,
                             avg_poor_tc, wiki_by_id, results_by_id)

    print("\nAll done")


if __name__ == "__main__":
    main()