@echo off
setlocal

set ENV_NAME=wiki

CALL %USERPROFILE%\anaconda3\Scripts\activate.bat
CALL conda activate %ENV_NAME%

set MODELS=mistral-7b_v01
set RAG_METHODS=baseline_rag naive_rag advanced_rag
set BUCKETS=good average poor

for %%M in (%MODELS%) do (
    echo Running model %%M...
    for %%R in (%RAG_METHODS%) do (
        echo Running method %%R...
        for %%B in (%BUCKETS%) do (
            echo Running bucket %%B...
            python .\rag_eval.py "%%M" "%%R" "%%B"
        )
    )
)

CALL conda deactivate

endlocal