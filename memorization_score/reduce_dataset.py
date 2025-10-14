import sys
import os
import json
from src.models import ModelManager
from src.processor import DataProcessor

MODEL_NAME = sys.argv[1]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "extracted_wikipedia_full.json")
OUTPUT_FILE =  os.path.join(BASE_DIR, "dataset", "reduced_by_token_count", f"extracted_wikipedia_{MODEL_NAME}_token_limit.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_NAME)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

model_manager = ModelManager(model_path=MODEL_PATH)
processor = DataProcessor(MODEL_NAME, model_manager)

reduced_data = processor.reduce_data_by_token_size(dataset)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(reduced_data, f, indent=4)

print(f"Results saved to {OUTPUT_FILE}")