import json
import sys
import os
import pandas as pd
from pprint import pprint
from scipy.stats import binomtest
from openpyxl import Workbook

MODEL_NAME = sys.argv[1]
RAG_TYPE = sys.argv[2]
BUCKET_NAME_1 = sys.argv[3]
BUCKET_NAME_2 = sys.argv[4]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_JUDGE_RESULTS_DIR = os.path.join(BASE_DIR, "results_llm_retrival_reranker_llama_3_3")
LLM_JUDGE_RESULTS_MODEL_DIR = os.path.join(LLM_JUDGE_RESULTS_DIR, MODEL_NAME)

JUDGEMENTS_FILE_1 = f"judgements_{MODEL_NAME}_{RAG_TYPE}_{BUCKET_NAME_1}.json"
JUDGEMENTS_FILE_2 = f"judgements_{MODEL_NAME}_{RAG_TYPE}_{BUCKET_NAME_2}.json"
JUDGEMENTS_RESULTS_1 = os.path.join(LLM_JUDGE_RESULTS_MODEL_DIR, JUDGEMENTS_FILE_1)
JUDGEMENTS_RESULTS_2 = os.path.join(LLM_JUDGE_RESULTS_MODEL_DIR, JUDGEMENTS_FILE_2)

OUTPUT_CSV = os.path.join(BASE_DIR, "statistic_test", "results", "statistic_results_binomal_greater.xlsx")

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_criterion(data, bucket_label):
    rows = []
    for item in data:
        for j in item['judgements']:
            for criterion in ['correctness', 'relevance', 'completeness']:
                if criterion not in j:
                    print(f"Missing '{criterion}' in judgement: {j}")
                rows.append({
                    'bucket': bucket_label,
                    'question_type': j['question_type'],
                    criterion: j[criterion]
                })
    return rows

def count_criterion_by_question_type(rows, criterion):
    """
    Returns a dict: {question_type: (successes, total)}
    """
    stats = {}
    question_types = set(row['question_type'] for row in rows)
    for qtype in question_types:
        filtered = [row for row in rows if row['question_type'] == qtype and criterion in row]
        successes = sum(1 for row in filtered if row[criterion] == 1)
        total = len(filtered)
        stats[qtype] = (successes, total)
    return stats

def transform_stats_to_flat_dict(bucket, stats_dict, criterion):
    """
    Transforms stats for a single criterion into a list of flat dicts.
    """
    output = []
    for question_type, (successes, total) in stats_dict.items():
        output.append({
            'bucket': bucket,
            'question_type': question_type,
            'criterion': criterion,
            'successes': successes,
            'total': total,
            'successes_rate': successes / total,
        })
    return output

def get_flat_stats(criterion_bucket_1, bucket_name):
    all_flat_stats = []
    for criterion in ['correctness', 'relevance', 'completeness']:
        stats = count_criterion_by_question_type(criterion_bucket_1, criterion)
        flat_stats = transform_stats_to_flat_dict(bucket_name, stats, criterion)
        all_flat_stats.extend(flat_stats)
    return all_flat_stats

def combine_bucket_stats(all_flat_stats_bucket_1, all_flat_stats_bucket_2):
    """
    Combines stats from two buckets, matching by question_type and criterion.
    Output keys use bucket names as prefixes (e.g., good_successes, poor_successes).
    Adds a 'buckets' field, e.g., 'good_poor'.
    """
    combined = []
    lookup_1 = {(d['question_type'], d['criterion']): d for d in all_flat_stats_bucket_1}
    lookup_2 = {(d['question_type'], d['criterion']): d for d in all_flat_stats_bucket_2}
    
    for key in set(lookup_1) & set(lookup_2):
        d1 = lookup_1[key]
        d2 = lookup_2[key]
        combined.append({
            'model': f"{MODEL_NAME}",
            'rag_type': f"{RAG_TYPE}",
            'buckets': f"{BUCKET_NAME_1}_{BUCKET_NAME_2}",
            'question_type': key[0],
            'criterion': key[1],
            f"bucket_1_successes": d1['successes'],
            f"bucket_1_total": d1['total'],
            f"bucket_1_successes_ratio": d1['successes_rate'],
            f"bucket_2_successes": d2['successes'],
            f"bucket_2_total": d2['total'],
            f"bucket_2_successes_ratio": d2['successes_rate'],
        })

    combined_df = transform_to_df(combined)
    return combined_df

def transform_to_df(combined_stats):
    column_order = [
        'model', 'rag_type', 'question_type', 'criterion', 'buckets',
        f'bucket_1_successes',
        f'bucket_1_total',
        f'bucket_1_successes_ratio',
        f'bucket_2_successes',
        f'bucket_2_total',
        f'bucket_2_successes_ratio'
    ]
    df = pd.DataFrame(combined_stats)
    df = df[column_order]
    return df

def perform_test(combined_stats):
    combined_stats['alternative'] = ""
    combined_stats['statistic'] = 0.0
    combined_stats['pvalue'] = 0.0

    for idx, row in combined_stats.iterrows():
        successes = row[f"bucket_1_successes"]
        n = row[f"bucket_1_total"]
        p_null = row[f"bucket_2_successes_ratio"]
        if n == 0:
            combined_stats.at[idx, 'alternative'] = None
            combined_stats.at[idx, 'statistic'] = None
            combined_stats.at[idx, 'pvalue'] = None
            continue
        result = binomtest(successes, n, p_null, alternative='greater')
        combined_stats.at[idx, 'alternative'] = result.alternative
        combined_stats.at[idx, 'statistic'] = result.statistic
        combined_stats.at[idx, 'pvalue'] = result.pvalue

    return combined_stats


def main():
    data_bucket_1 = load_data(JUDGEMENTS_RESULTS_1)
    data_bucket_2 = load_data(JUDGEMENTS_RESULTS_2)

    criterion_bucket_1 = get_criterion(data_bucket_1, BUCKET_NAME_1)
    criterion_bucket_2 = get_criterion(data_bucket_2, BUCKET_NAME_2)

    all_flat_stats_bucket_1 = get_flat_stats(criterion_bucket_1, BUCKET_NAME_1)
    all_flat_stats_bucket_2 = get_flat_stats(criterion_bucket_2, BUCKET_NAME_2)

    combined_stats = combine_bucket_stats(
        all_flat_stats_bucket_1,
        all_flat_stats_bucket_2
    )

    results = perform_test(combined_stats)

    if os.path.exists(OUTPUT_CSV):
        prev_results = pd.read_excel(OUTPUT_CSV)
        combined_df = pd.concat([prev_results, results], ignore_index=True)
        combined_df.to_excel(OUTPUT_CSV, index=False)
    else:
        results.to_excel(OUTPUT_CSV)

    print(f"Saved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
