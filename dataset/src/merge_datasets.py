import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEPSEEK_DATASET_PATH = os.path.join(BASE_DIR, "reduced_by_token_count", "extracted_wikipedia_deepseek_llm_7b_token_limit.json")
LLAMA_DATASET_PATH = os.path.join(BASE_DIR, "reduced_by_token_count", "extracted_wikipedia_llama_2_7b_hf_token_limit.json")
MISTRAL_DATASET_PATH = os.path.join(BASE_DIR, "reduced_by_token_count", "extracted_wikipedia_mistral-7b_v01_token_limit.json")
OUTPUT_PATH =  os.path.join(BASE_DIR, "extracted_wikipedia_token_limit.json")

with open(DEEPSEEK_DATASET_PATH, "r", encoding="utf-8") as f:
    deepseek_dataset = json.load(f)
with open(LLAMA_DATASET_PATH, "r", encoding="utf-8") as f:
    llama_dataset = json.load(f)
with open(MISTRAL_DATASET_PATH, "r", encoding="utf-8") as f:
    mistral_dataset = json.load(f)

deepseek_dict = {item['id']: item for item in deepseek_dataset}
llama_dict = {item['id']: item for item in llama_dataset}
mistral_dict = {item['id']: item for item in mistral_dataset}

common_ids = set(deepseek_dict.keys()) & set(llama_dict.keys()) & set(mistral_dict.keys())
final_dataset = [deepseek_dict[i] for i in sorted(common_ids)]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, indent=4)

print(f"Final dataset contains {len(final_dataset)} items.")
print(f"Results saved to {OUTPUT_PATH}")