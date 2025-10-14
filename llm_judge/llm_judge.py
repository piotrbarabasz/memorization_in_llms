import os
import json
import requests
import sys
import re
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()
API_TOKEN = os.getenv("CLARIN_API_TOKEN")
API_URL = "https://services.clarin-pl.eu/api/v1/oapi/chat/completions"

MODEL_NAME = sys.argv[1]
RAG_TYPE = sys.argv[2]
BUCKET_NAME = sys.argv[3]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAG_RESULTS_DIR = os.path.join(BASE_DIR, "rag", "results_llm_retrival_reranker")
RAG_RESULTS_MODEL_PATH = os.path.join(RAG_RESULTS_DIR, MODEL_NAME)

JSON_FILE = f"{MODEL_NAME}_{BUCKET_NAME}_{RAG_TYPE}.json"
RAG_RESULT_MODEL_FILE_PATH = os.path.join(RAG_RESULTS_MODEL_PATH, JSON_FILE)

OUTPUT_FILE = f"judgements_{MODEL_NAME}_{RAG_TYPE}_{BUCKET_NAME}.json"
OUTPUT_RESULTS_DIR = os.path.join(BASE_DIR, "llm_judge", "results_llm_retrival_reranker_llama_3_3")
OUTPUT_FILE_PATH = os.path.join(OUTPUT_RESULTS_DIR, MODEL_NAME, OUTPUT_FILE)

if not os.path.isfile(RAG_RESULT_MODEL_FILE_PATH):
    print(f"[ERROR] File does not exist: {RAG_RESULT_MODEL_FILE_PATH}")
    sys.exit(0)

def load_questions(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_results(results_path):
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            processed_results = json.load(f)
    else:
        processed_results = []

    print(len(processed_results))
    return processed_results

def filter_data(all_questions, processed_results):
    processed_keys = {
        (int(entry["id"]), entry["rag_method"])
        for entry in processed_results
    }

    filtered_questions = [
        q for q in all_questions
        if (int(q["id"]), q["rag_method"]) not in processed_keys
    ]

    return filtered_questions

def llm_evaluate(prompt):
    headers = {
        'Content-Type': 'application/json', 
        'api-token': API_TOKEN
        }
    data = {
        "model": "llama3.3", 
        "messages": 
            [
                {
                    "role": "system", 
                    "content": "You are a helpful, accurate, and critical judge."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        print("Status Code:", response.status_code)

        if response.status_code != 200:
            print("Request failed:", response.text)
            raise RuntimeError("API response error")

        full_response = response.json()

        content = full_response['choices'][0]['message']['content'].strip()
        print("content", content)

        match = re.search(r'```json\\s*(\\{.*\\})\\s*```', content, re.DOTALL)
        if match:
            content = match.group(1)

        return json.loads(content)

    except Exception as e:
        print("Exception in llm_evaluate:", e)
        raise

def main():
    all_questions = load_questions(RAG_RESULT_MODEL_FILE_PATH)
    processed_results = load_results(OUTPUT_FILE_PATH)
    all_entries = filter_data(all_questions, processed_results)

    final_results = processed_results
    # all_entries = all_entries[:2]
    for idx, entry in enumerate(all_entries, 1):
        print(f"[PROCESSING] ({idx}/{len(all_entries)}) ID: {entry['id']}, with method: {entry['rag_method']}")
        judgements = []
        question_types = ["comprehension", "analytical", "textual_stylistic"]

        for q_idx, q_type in enumerate(question_types, 1):
            prompt = f"""
            You are an expert educational evaluator. For the given question and answers, evaluate the **generated answer** for each of the following criteria **independently**:
            - Correctness (is the answer factually correct and accurate?)
            - Relevance (is the answer relevant to the question?)
            - Completeness (does the answer fully address all aspects of the question?)

            Return your judgement as a JSON object with three keys: "correctness", "relevance", "completeness". Each value should be **1 (yes)** or **0 (no)**.
            Do not include any extra text like ```python ...```.

            Example response:
            {{
                "correctness": 1,
                "relevance": 0,
                "completeness": 1
            }}

            Question Type: {q_type}
            Question: {entry[q_type + '_question']}
            Original Answer: {entry['original_' + q_type + '_answer']}
            Generated Answer: {entry['generated_' + q_type + '_answer']}
            """

            print(f"[PROCESSING] ({q_idx}/3) Question type: {q_type}")
            judgement = llm_evaluate(prompt)
            judgement['question_type'] = q_type
            judgements.append(judgement)
            print(f"[DONE] ({q_idx}/3) {judgement}")

        result_entry = {
            "id": entry["id"],
            "model": entry["model"],
            "rag_method": entry["rag_method"],
            "bucket": entry["bucket"],
            "judgements": judgements
        }
        final_results.append(result_entry)

        with open(OUTPUT_FILE_PATH, "w", encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"[SAVED] ({idx}/{len(all_entries)}) ID: {entry['id']}")

    print("Evaluation completed and saved.")

if __name__ == "__main__":
    main()
