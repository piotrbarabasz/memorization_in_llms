import json
import os
import sys
from collections import OrderedDict
from pprint import pprint

MODEL_NAME = sys.argv[1]
RAG_TYPE = sys.argv[2]
BUCKET_NAME = sys.argv[3]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAG_RESULTS_DIR = os.path.join(BASE_DIR, "rag", "results_new_rag_fixed")
RAG_RESULTS_MODEL_PATH = os.path.join(RAG_RESULTS_DIR, MODEL_NAME)

JSON_FILE = f"{MODEL_NAME}_{BUCKET_NAME}_{RAG_TYPE}.json"
JSON_FILE_PATH = os.path.join(RAG_RESULTS_MODEL_PATH, JSON_FILE)

if not os.path.isfile(JSON_FILE_PATH):
    print(f"[ERROR] File does not exist: {JSON_FILE_PATH}")
    sys.exit(0)

with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

ordered_data = []
for entry in data:
    rag_method = entry.get("rag_method")
    if isinstance(rag_method, dict):
        rag_method = rag_method.get("method", "unknown")

    new_entry = OrderedDict()
    after_rag_method = False
    for key, value in entry.items():
        if key == "rag_method":
            new_entry[key] = rag_method
            new_entry["bucket"] = BUCKET_NAME
            after_rag_method = True
        elif key == "bucket":
            continue
        else:
            new_entry[key] = value
    if not after_rag_method:
        new_entry["bucket"] = BUCKET_NAME
    ordered_data.append(new_entry)

pprint(list(ordered_data)[:2])

with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
    json.dump(ordered_data, f, indent=2, ensure_ascii=False)

print(f"Updated {len(ordered_data)} entries and saved to {JSON_FILE_PATH}.")

"""
Run with:

python .\llm_judge\src\add_bucket_to_data.py deepseek_llm_7b naive_rag poor

Available models:

    deepseek_llm_7b
    llama_2_7b_hf
    mistral-7b_v01

Available Rag Methods:

    naive_rag
    advanced_rag
    modular_rag

Memorization buckets:

    good
    average
    poor
"""