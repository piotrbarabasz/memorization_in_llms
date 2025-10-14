import json
import os
import random

N_RECORDS = 5000
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "extracted_wikipedia_token_limit.json")
OUTPUT_PATH = os.path.join(BASE_DIR, f"extracted_wikipedia_token_limit_records_{N_RECORDS}.json")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

id_to_data = {entry['id']: entry for entry in dataset}
random_ids = random.sample(list(id_to_data.keys()), N_RECORDS)
subset = [id_to_data[rid] for rid in random_ids]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(subset, f, indent=4, ensure_ascii=False)

print(f"Subset of {N_RECORDS} records saved to {OUTPUT_PATH}")
