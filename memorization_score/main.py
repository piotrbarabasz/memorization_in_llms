import sys
import os
import json

from src.models import ModelManager
from src.processor import DataProcessor

MODEL_NAME = sys.argv[1]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "extracted_wikipedia_token_limit_records_5000.json")
OUTPUT_FILE =  os.path.join(BASE_DIR, "memorization_score", "results", f"{MODEL_NAME}_perplexity.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_NAME)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

model_manager = ModelManager(model_path=MODEL_PATH)
processor = DataProcessor(MODEL_NAME, model_manager)
# N_RECORDS = 2

# output = processor.process_data(dataset[:N_RECORDS])
output = processor.process_data(dataset)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print(f"Results saved to {OUTPUT_FILE}")