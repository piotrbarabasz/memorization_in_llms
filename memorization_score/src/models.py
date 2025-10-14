import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ModelManager:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = self.load_tokenizer(model_path)
        self.model = self.load_model(model_path)

        print(f"Working on {self.device}")

    def load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        return tokenizer

    def load_model(self, model_path):
        quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            quantization_config=quant_config,
            low_cpu_mem_usage=True
        )
        return model