import os
import json

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_existing_results(file_path):
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data

    return []