import torch
import gc
import time
import os
import json
import time

from collections import Counter
from src.metrics import ModelMetrics

class DataProcessor:
    def __init__(self, model_name, model_manager):
        self.model_name = model_name
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

    def compute_global_avg_token_count(self, input_data):
        total_tokens = 0
        total_entries = 0

        for item in input_data:
            text = item["text"]
            total_tokens += ModelMetrics.compute_token_count(self.tokenizer, text)
            total_entries += 1

        return total_tokens / total_entries if total_entries > 0 else 1
    
    def compute_token_frequencies(self, input_data):
        token_counter = Counter()
        for item in input_data:
            tokens = self.tokenizer.encode(item["text"], truncation=True, max_length=1024)
            token_counter.update(tokens)
        return token_counter
    
    def compute_rare_token_ratio(self, text, token_counter, rare_threshold=5):
        tokens = self.tokenizer.encode(text, truncation=True, max_length=1024)
        if not tokens:
            return 0.0
        rare_count = sum(1 for t in tokens if token_counter[t] <= rare_threshold)
        return rare_count / len(tokens)
    
    def reduce_data_by_token_size(self, input_data):
        reduced_data = []
        for item in input_data:
            print(f"[Processing] Article title: {item['title']} with id: {item['id']}.")
            token_count = ModelMetrics.compute_token_count(self.tokenizer, item["text"])
            if token_count > 200:
                reduced_data.append(item)
            else:
                print(f"[Dropping] ID: {item['id']} with token count: {token_count}.")
        return reduced_data
    
    def get_response_template(self, dataset_size, avg_token_count):
        response = {
            "metadata": {
                "model": self.model_name,
                "dataset_size": dataset_size,
                "avg_token_count": avg_token_count,
                "execution_time": None
            },
            "results": []
        }
        return response
    
    def get_result_template(self, item):
        result = {
                "id": item["id"],
                "category": item["category"],
                "title": item["title"]
            }
        return result

    def process_data(self, input_data):
        start_time = time.time()  

        avg_token_count = self.compute_global_avg_token_count(input_data)
        token_counter = self.compute_token_frequencies(input_data)

        response = self.get_response_template(len(input_data), avg_token_count)

        for item in input_data:
            res_entry = self.get_result_template(item)

            print(f"[Processing] Article title: {res_entry['title']} with id: {res_entry['id']}.")

            token_count = ModelMetrics.compute_token_count(self.tokenizer, item["text"])
            perplexity = ModelMetrics.compute_perplexity(self.model, self.tokenizer, item["text"], next(self.model.parameters()).device)
            adjusted_perplexity = ModelMetrics.compute_adjusted_perplexity(
                perplexity,
                token_count,
                avg_token_count
            )

            rare_token_ratio = self.compute_rare_token_ratio(item["text"], token_counter)
            
            res_entry["results"] = {
                "perplexity": round(perplexity, 2),
                "token_count": token_count,
                "adjusted_perplexity": round(adjusted_perplexity, 2),
                "rare_token_ratio": round(rare_token_ratio, 2)
            }

            response["results"].append(res_entry)
        response["metadata"]["execution_time"] = round(time.time() - start_time, 2)
        return response

    # def process_and_save_batches(self, dataset, output_file, save_every=10):
    #     # Load existing results if any
    #     already_processed_ids = set()
    #     if os.path.exists(output_file):
    #         with open(output_file, "r", encoding="utf-8") as f:
    #             existing_output = json.load(f)
    #         already_processed_ids = {entry["id"] for entry in existing_output.get("results", [])}
    #         print(f"[INFO] Loaded {len(already_processed_ids)} already processed entries.")
    #     else:
    #         existing_output = {
    #             "metadata": {
    #                 "dataset_size": len(dataset),
    #                 "models": list(self.models.keys()),
    #                 "timestamp": None
    #             },
    #             "results": []
    #         }

    #     # Filter out already-processed records
    #     unprocessed_data = [record for record in dataset if record["id"] not in already_processed_ids]
    #     print(f"[INFO] Found {len(unprocessed_data)} new records to process.")

    #     global_avg_token_count = self.compute_global_avg_token_count(unprocessed_data)
    #     # Process in batches
    #     for i in range(0, len(unprocessed_data), save_every):
    #         batch = unprocessed_data[i:i + save_every]
    #         # output_batch = self.process_data(batch)
    #         output_batch = self.process_data(batch, global_avg_token_count)
    #         existing_output["results"].extend(output_batch["results"])
    #         existing_output["metadata"]["timestamp"] = output_batch["metadata"]["timestamp"]

    #         with open(output_file, "w", encoding="utf-8") as f:
    #             json.dump(existing_output, f, indent=4)

    #         print(f"[INFO] Saved batch {i // save_every + 1}, total processed: {len(existing_output['results'])}")

    #     print(f"Done. Results saved to {output_file}")
