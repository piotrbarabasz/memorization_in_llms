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

if not os.path.isfile(JSON_FILE_PATH):
    print(f"[ERROR] File does not exist: {JSON_FILE_PATH}")
    sys.exit(0)

with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

def extract_answer(text):
    marker = "?\n"
    idx = text.find(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()
    else:
        return text.strip()
    
for entry in data:
    for key in [
        "generated_comprehension_answer",
        "generated_analytical_answer",
        "generated_textual_stylistic_answer"
    ]:
        if key in entry:
            entry[key] = extract_answer(entry[key])

OUTPUT_FILE=f"{MODEL_NAME}_{BUCKET_NAME}_{RAG_TYPE}.json"
OUTPUT_RAG_RESULTS_DIR = os.path.join(BASE_DIR, "rag", "results_new_rag_fixed_copy")
OUTPUT_RAG_RESULTS_MODEL_PATH = os.path.join(OUTPUT_RAG_RESULTS_DIR, MODEL_NAME)
OUTPUT_PATH=os.path.join(OUTPUT_RAG_RESULTS_MODEL_PATH, OUTPUT_FILE) 

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)