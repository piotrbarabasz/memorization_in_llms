import os
from datetime import datetime

from src.utils import load_dataset, load_existing_results
from src.llm_api_questions import get_questions
from src.result_writer import save_results_with_metadata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "extracted_wikipedia_token_limit_records_5000.json")
RESULTS_DIR = os.path.join(BASE_DIR, "generate_questions", "results")

def main():
    dataset = load_dataset(DATASET_PATH)
    dataset_name = os.path.basename(DATASET_PATH)
    dataset_size = len(dataset)
    timestamp = datetime.now().strftime("%Y_%m_%d")
    output_path = os.path.join(RESULTS_DIR, f"questions_{timestamp}.json")
    all_results = load_existing_results(output_path)
    existing_ids = {entry["id"] for entry in all_results}

    print(f"Loaded {len(existing_ids)} previously processed entries. Resuming...")

    for idx, item in enumerate(dataset, start=1):
        item_id = item["id"]
        text = item["text"]
        
        if item_id in existing_ids:
            print(f"[{item_id}] Skipped (already processed)")
            continue

        print(f"Processing ID: {item_id}")
        result = get_questions(text)

        if result:
            ordered_result = {"id": item_id}
            ordered_result.update(result)
            all_results.append(ordered_result)
            print(f"[{item_id}] Success")

            if len(all_results) % 10 == 0:
                save_results_with_metadata(output_path, all_results, dataset_name, dataset_size)
                print(f"Progress saved after {len(all_results)} entries.")

    save_results_with_metadata(output_path, all_results, dataset_name, dataset_size)
    print(f"\nAll results saved to {output_path}")

if __name__ == "__main__":
    main()
