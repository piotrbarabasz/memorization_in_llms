import os
import sys
import json
from pprint import pprint

MODEL_NAME = sys.argv[1]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAG_RESULTS_DIR = os.path.join(BASE_DIR, "rag", "results")
RAG_RESULTS_MODEL_PATH = os.path.join(RAG_RESULTS_DIR, MODEL_NAME)

def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                data.extend(file_data)
    return data

if __name__ == "__main__":
    model_results = load_json_files(RAG_RESULTS_MODEL_PATH)
    pprint(model_results)
    pprint(len(model_results))