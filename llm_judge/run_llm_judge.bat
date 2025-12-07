@echo off
setlocal

set ENV_NAME=wiki

CALL %USERPROFILE%\anaconda3\Scripts\activate.bat
CALL conda activate %ENV_NAME%

set MODELS=deepseek_llm_7b llama_2_7b_hf mistral-7b_v01
set RAG_METHODS=baseline_rag naive_rag advanced_rag
set BUCKETS=good average poor

for %%M in (%MODELS%) do (
    for %%R in (%RAG_METHODS%) do (
        for %%B in (%BUCKETS%) do (
            echo Running model %%M...
            echo Running method %%R...
            echo Running bucket %%B...
            python .\llm_judge.py "%%M" "%%R" "%%B"
        )
    )
)

CALL conda deactivate

echo.
echo ===== All done! =====
pause

endlocal