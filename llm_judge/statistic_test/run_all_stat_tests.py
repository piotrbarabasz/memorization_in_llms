import os
import sys
import itertools

models = ['deepseek_llm_7b', 'llama_2_7b_hf', 'mistral-7b_v01']
rag_types = ['baseline_rag', 'naive_rag', 'advanced_rag']
buckets = ['good', 'average', 'poor']
comparisons = [('good', 'average'), ('good', 'poor'), ('average', 'poor')]

# SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "statistic_test_two_files_additive_model.py")
# SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "statistic_test_two_files_interaction_model.py")
# SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "statistic_test_two_files_combined.py")
# SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "statistic_test_two_files_fisher.py")
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "statistic_test_two_files_binomal.py")

for model in models:
    for rag_type in rag_types:
        for bucket1, bucket2 in comparisons:
            print(f"\nRunning: {model}, {rag_type}, {bucket1} vs {bucket2}")
            exit_code = os.system(
                f'python "{SCRIPT_PATH}" "{model}" "{rag_type}" "{bucket1}" "{bucket2}"'
            )
            if exit_code != 0:
                print(f"Failed: {model}, {rag_type}, {bucket1} vs {bucket2}")
