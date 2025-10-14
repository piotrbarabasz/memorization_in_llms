from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
# from huggingface_hub import login
# import os
# from dotenv import load_dotenv

# load_dotenv()

# HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# login(HUGGINGFACE_TOKEN)

def load_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(model_name, device=device)

def load_embedder(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# def load_generator(model_name="t5-small", device=-1):
#     return pipeline("text2text-generation", model=model_name, device=device)

def load_generator(model_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        quantization_config=quant_config,
        low_cpu_mem_usage=True

    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True
    )
