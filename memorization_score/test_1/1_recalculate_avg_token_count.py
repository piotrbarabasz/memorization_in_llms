import os
import json
import statistics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
BUCKETS_DIR = os.path.join(BASE_DIR, "results", "buckets_100_samples_avg_poor")

MODELS = [
    "deepseek_llm_7b",
    "llama_2_7b_hf",
    "mistral-7b_v01",
]

BUCKETS = ["average", "good", "poor"]

def recalc_avg_token_count(json_path: str, bucket: str):
    """Recalculate and overwrite avg_token_count_bucket_<bucket> in metadata."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    token_counts = [item["token_count"] for item in data["results"]]
    avg_token_count = statistics.mean(token_counts)

    field_name = f"avg_token_count_bucket_{bucket}"
    if "metadata" not in data:
        data["metadata"] = {}

    data["metadata"][field_name] = avg_token_count

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] {os.path.basename(json_path)} → {field_name} = {avg_token_count:.4f}")


def main():
    for model in MODELS:
        for bucket in BUCKETS:
            filename = f"{model}_bucket_{bucket}.json"
            json_path = os.path.join(BUCKETS_DIR, model, filename)

            if not os.path.exists(json_path):
                print(f"[SKIP] Missing file: {json_path}")
                continue

            recalc_avg_token_count(json_path, bucket)


if __name__ == "__main__":
    main()

"""
[OK] deepseek_llm_7b_bucket_average.json → avg_token_count_bucket_average = 313.7000
[OK] deepseek_llm_7b_bucket_good.json → avg_token_count_bucket_good = 633.4600
[OK] deepseek_llm_7b_bucket_poor.json → avg_token_count_bucket_poor = 311.8900

[OK] llama_2_7b_hf_bucket_average.json → avg_token_count_bucket_average = 276.0200
[OK] llama_2_7b_hf_bucket_good.json → avg_token_count_bucket_good = 490.5200
[OK] llama_2_7b_hf_bucket_poor.json → avg_token_count_bucket_poor = 269.1800

[OK] mistral-7b_v01_bucket_average.json → avg_token_count_bucket_average = 275.3500
[OK] mistral-7b_v01_bucket_good.json → avg_token_count_bucket_good = 478.9000
[OK] mistral-7b_v01_bucket_poor.json → avg_token_count_bucket_poor = 266.8600
"""

