import json
import os
import torch
import gc

class Processor():
    def __init__(self, rag, model_name, output_path="rag/results/processed_results.json"):
        self.rag = rag
        self.model_name = model_name
        self.output_path = output_path

    def process(self, questions, dataset, bucket_name, rag_method='naive_rag', batch_size=10):
        results = []
        count = 0

        dataset_dict = {entry['id']: entry['text'] for entry in dataset}

        for item in questions:
            # torch.cuda.empty_cache()
            # gc.collect()

            context = dataset_dict[str(item['id'])]

            if not context:
                print(f"[SKIPPED] ID {item['id']} not found in dataset.")
                continue

            self.rag.build_index([context])
            print(f"[PROCESSING] ID {item['id']} ({count + 1}/{len(questions)})")

            result = {
                "id": item["id"],
                "model": self.model_name,
                "rag_method": rag_method,
                "bucket": bucket_name
            }

            question_fields = [
                ("comprehension_question", "comprehension_answer"),
                ("analytical_question", "analytical_answer"),
                ("textual_stylistic_question", "textual_stylistic_answer")
            ]

            for question_key, original_answer_key in question_fields:
                question_text = item.get(question_key)
                original_answer = item.get(original_answer_key) # or item.get(original_answer_key.replace("_answer", "_awnser"))  # fallback for typo
                if question_text:
                    result[question_key] = question_text
                    result[f"original_{original_answer_key}"] = original_answer
                    result[f"generated_{original_answer_key}"] = self.get_answer(question_text, rag_method)


            results.append(result)
            count += 1

            if count % batch_size == 0:
                self.save_batch(results)
                print(f"[SAVED] Batch of {batch_size} processed items.")
                results = []

        if results:
            self.save_batch(results)

    def get_answer(self, question_text, rag_method):
        raw = getattr(self.rag, rag_method)(question_text)

        if "### Answer:" in raw:
            answer_part = raw.split("### Answer:", 1)[1].lstrip()
        elif raw.lstrip().lower().startswith("answer:"):
            answer_part = raw.partition(":")[2].lstrip()
        else:
            answer_part = raw.strip()

        return answer_part

    def save_batch(self, results):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            existing.extend(results)
        else:
            existing = results

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
