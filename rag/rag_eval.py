import json
import sys
import os

from src.models import load_embedder, load_generator, load_reranker
from src.rag_new import RAG
from src.processor import Processor
from src.data_check import filter_questions_by_dataset_ids, sort_by_id

## TODO: add bucket type to response JSON

MODEL_NAME = sys.argv[1]
RAG_TYPE = sys.argv[2]
BUCKET = sys.argv[3] 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKETS_DIR = os.path.join(BASE_DIR, "memorization_score", "results", "buckets_100_samples_avg_poor")

DATASET_PATHS = {
    "deepseek_llm_7b": {
        "good": os.path.join(BUCKETS_DIR, "deepseek_llm_7b", "deepseek_llm_7b_bucket_good.json"),
        "average": os.path.join(BUCKETS_DIR, "deepseek_llm_7b", "deepseek_llm_7b_bucket_average.json"),
        "poor": os.path.join(BUCKETS_DIR, "deepseek_llm_7b", "deepseek_llm_7b_bucket_poor.json")
    },
    "llama_2_7b_hf": {
        "good": os.path.join(BUCKETS_DIR, "llama_2_7b_hf", "llama_2_7b_hf_bucket_good.json"),
        "average": os.path.join(BUCKETS_DIR, "llama_2_7b_hf", "llama_2_7b_hf_bucket_average.json"),
        "poor": os.path.join(BUCKETS_DIR, "llama_2_7b_hf", "llama_2_7b_hf_bucket_poor.json")
    },
    "mistral-7b_v01": {
        "good": os.path.join(BUCKETS_DIR, "mistral-7b_v01", "mistral-7b_v01_bucket_good.json"),
        "average": os.path.join(BUCKETS_DIR, "mistral-7b_v01", "mistral-7b_v01_bucket_average.json"),
        "poor": os.path.join(BUCKETS_DIR, "mistral-7b_v01", "mistral-7b_v01_bucket_poor.json")
    }
}

QUESTIONS_PATH = os.path.join(BASE_DIR, "generate_questions", "results", "questions_2025_05_27.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "rag", "results_llm_retrival_reranker", f"{MODEL_NAME}_{BUCKET}_{RAG_TYPE}.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_NAME)

DATASET_PATH = DATASET_PATHS[MODEL_NAME][BUCKET]

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    dataset = json.load(f)["results"]

with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
    questions = json.load(f)["results"]

filtered_questions = filter_questions_by_dataset_ids(dataset, questions)
sorted_dataset = sort_by_id(dataset)
sorted_filtered_questions = sort_by_id(filtered_questions)

embedder = load_embedder("all-MiniLM-L6-v2")
generator = load_generator(MODEL_PATH)
reranker = load_reranker() 

# rag_system = RAG(embedder, generator)
rag_system = RAG(embedder, generator, reranker=reranker)
processor = Processor(rag_system, model_name=MODEL_NAME, output_path=OUTPUT_PATH)

processor.process(
    questions=sorted_filtered_questions,
    dataset=sorted_dataset,
    bucket_name=BUCKET,
    rag_method=RAG_TYPE,
    batch_size=1
)
