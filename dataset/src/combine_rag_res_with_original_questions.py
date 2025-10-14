import json
import os
import sys

MODEL_NAME = sys.argv[1]
RAG_TYPE = sys.argv[2]
BUCKET_NAME = sys.argv[3]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAG_RESULTS_DIR = os.path.join(BASE_DIR, "rag", "results_new_rag_fixed_copy")
RAG_RESULTS_MODEL_PATH = os.path.join(RAG_RESULTS_DIR, MODEL_NAME)

JSON_FILE = f"{MODEL_NAME}_{BUCKET_NAME}_{RAG_TYPE}.json"
JSON_FILE_PATH = os.path.join(RAG_RESULTS_MODEL_PATH, JSON_FILE)

WIKI_PATH = "C:/Users/user/Desktop/MSc/pb_msc/dataset/extracted_wikipedia_token_limit_records_5000.json"


OUTPUT_FILE=f"{MODEL_NAME}_{BUCKET_NAME}_{RAG_TYPE}.json"
OUTPUT_RAG_RESULTS_DIR = os.path.join(BASE_DIR, "rag", "results_question_with_answers")
OUTPUT_RAG_RESULTS_MODEL_PATH = os.path.join(OUTPUT_RAG_RESULTS_DIR, MODEL_NAME)
OUTPUT_PATH=os.path.join(OUTPUT_RAG_RESULTS_MODEL_PATH, OUTPUT_FILE) 

if not os.path.isfile(JSON_FILE_PATH):
    print(f"[ERROR] File does not exist: {JSON_FILE_PATH}")
    sys.exit(0)

with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Step 1. Load Wikipedia data
with open(WIKI_PATH, "r", encoding="utf-8") as f:
    wiki_data = json.load(f)

# Step 2. Extract set of IDs from deepseek
deepseek_ids = {item["id"] for item in data}

# Step 3. Create ID -> Wikipedia record map
wiki_map = {item["id"]: item for item in wiki_data if item["id"] in deepseek_ids}

# Step 4. Merge matched items
merged_data = []

for item in data:
    wiki = wiki_map.get(item["id"])
    if wiki:
        merged_record = {
            "id": item["id"],
            "model": item["model"],
            "rag_method": item["rag_method"],
            "bucket": item["bucket"],
            "category": wiki.get("category", ""),
            "title": wiki.get("title", ""),
            "text": wiki.get("text", ""),
            "comprehension_question": item.get("comprehension_question", ""),
            "original_comprehension_answer": item.get("original_comprehension_answer", ""),
            "generated_comprehension_answer": item.get("generated_comprehension_answer", ""),
            "analytical_question": item.get("analytical_question", ""),
            "original_analytical_answer": item.get("original_analytical_answer", ""),
            "textual_stylistic_question": item.get("textual_stylistic_question", ""),
            "original_textual_stylistic_answer": item.get("original_textual_stylistic_answer", ""),
            "generated_textual_stylistic_answer": item.get("generated_textual_stylistic_answer", "")
        }
        merged_data.append(merged_record)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"Merged {len(merged_data)} records into {OUTPUT_PATH}")
