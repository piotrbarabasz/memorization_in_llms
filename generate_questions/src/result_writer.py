import json

def save_results_with_metadata(output_path, all_results, dataset_name, dataset_size):
    metadata = {
        "source_dataset": dataset_name,
        "source_dataset_size": str(dataset_size),
        "question_dataset_size": str(len(all_results))
    }
    save_data = {
        "metadata": metadata,
        "results": all_results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)