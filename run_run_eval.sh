#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="msc"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

conda activate "$ENV_NAME"


MODELS=(deepseek_llm_7b llama_2_7b_hf mistral-7b_v01)
RAG_METHODS=(advanced_rag)
BUCKETS=(good average poor)

for M in "${MODELS[@]}"; do
  echo "Running model $M..."
  for R in "${RAG_METHODS[@]}"; do
    echo "  Running method $R..."
    for B in "${BUCKETS[@]}"; do
      echo "    Running bucket $B..."
      python ./rag_eval.py "$M" "$R" "$B"
    done
  done
done

conda deactivate