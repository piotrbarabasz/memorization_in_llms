import json
import os
import random

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "memorization_score", "results")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "extracted_wikipedia_token_limit_records_5000.json")
MODEL_FILES = {
    "deepseek_llm_7b": os.path.join(RESULTS_DIR, "deepseek_llm_7b_perplexity.json"),
    "llama_2_7b_hf": os.path.join(RESULTS_DIR, "llama_2_7b_hf_perplexity.json"),
    "mistral-7b_v01": os.path.join(RESULTS_DIR, "mistral-7b_v01_perplexity.json"),
}

BUCKETS_DIR = os.path.join(RESULTS_DIR, "buckets")
os.makedirs(BUCKETS_DIR, exist_ok=True)

def bucket_label(adj_ppl):
    if adj_ppl < 10:
        return "good"
    elif adj_ppl < 29.5:
        return "average"
    else:
        return "poor"

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    wiki_data = json.load(f)
wiki_by_id = {item["id"]: item for item in wiki_data}

for model_name, model_path in MODEL_FILES.items():
    with open(model_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    results_by_id = {item["id"]: item["results"] for item in model_data["results"]}

    buckets = {"good": [], "average": [], "poor": []}

    for _id, results in results_by_id.items():
        adj_ppl = results.get("adjusted_perplexity")
        if adj_ppl is not None:
            bucket = bucket_label(adj_ppl)
            buckets[bucket].append(_id)

    for bucket in ["good", "average", "poor"]:
        ids = buckets[bucket]
        if len(ids) == 0:
            print(f"[{model_name}] No samples in bucket '{bucket}'. Skipping.")
            continue
        n_samples = min(100, len(ids))
        sampled_ids = random.sample(ids, n_samples)

        output = {
            "metadata": {
                "model": model_name,
                "bucket_bin": bucket,
                "dataset_size": n_samples
            },
            "results": []
        }

        for _id in sampled_ids:
            wiki = wiki_by_id.get(_id)
            results = results_by_id.get(_id)
            if wiki and results:
                entry = {
                    "id": str(_id),
                    "adjusted_perplexity": str(results.get("adjusted_perplexity")),
                    "token_count": str(results.get("token_count")),
                    "rare_token_ratio": str(results.get("rare_token_ratio")),
                    "category": wiki.get("category", ""),
                    "title": wiki.get("title", ""),
                    "text": wiki.get("text", ""),
                }
                output["results"].append(entry)

        out_dir = os.path.join(BUCKETS_DIR, model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{model_name}_bucket_{bucket}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(output['results'])} samples to {out_path}")
