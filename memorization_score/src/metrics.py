import torch
import math

class ModelMetrics:
    @staticmethod
    def compute_perplexity(model, tokenizer, text, device):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        perplexity = math.exp(loss)
        return perplexity

    @staticmethod
    def compute_token_count(tokenizer, text):
        return len(tokenizer.encode(text, truncation=True, max_length=1024))

    @staticmethod
    def compute_adjusted_perplexity(perplexity, token_count, avg_token_count):
        if token_count == 0:
            return float('inf')
        return perplexity * (avg_token_count / token_count)
