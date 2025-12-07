import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(BASE_DIR, "results", "llm_judge_statistic_results_new_rag_binomal_llama_3_3.csv")
COMPARED_RAG_TYPES_RESULTS = os.path.join(BASE_DIR, "results", "llm_judge_statistic_results_new_rag_binomal_comparison.csv")

df = pd.read_csv(RESULT_FILE)

pivot_stat = df.pivot_table(
    index=['model', 'question_type', 'criterion'],
    columns='rag_type',
    values='statistic'
).reset_index()

pivot_pval = df.pivot_table(
    index=['model', 'question_type', 'criterion'],
    columns='rag_type',
    values='pvalue'
).reset_index()

comparison = pivot_stat.copy()
for col in [c for c in pivot_pval.columns if c not in comparison.columns or c in ['model', 'question_type', 'criterion']]:
    if col not in ['model', 'question_type', 'criterion']:
        comparison[f'pvalue_{col}'] = pivot_pval[col]

comparison.to_csv(COMPARED_RAG_TYPES_RESULTS, index=False)

print(f"Comparison CSV saved in: {COMPARED_RAG_TYPES_RESULTS}")

print(comparison)
