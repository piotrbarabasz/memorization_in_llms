Available models:

    deepseek_llm_7b
    llama_2_7b_hf
    mistral-7b_v01

Available Rag Methods:

    naive_rag
    advanced_rag
    modular_rag

Memorization buckets:

    good
    average
    poor
    


python rag_eval.py deepseek_llm_7b naive_rag good +
python rag_eval.py deepseek_llm_7b naive_rag average +
python rag_eval.py deepseek_llm_7b naive_rag poor +

python rag_eval.py deepseek_llm_7b advanced_rag good +
python rag_eval.py deepseek_llm_7b advanced_rag average +
python rag_eval.py deepseek_llm_7b advanced_rag poor +

python rag_eval.py deepseek_llm_7b modular_rag good + 
python rag_eval.py deepseek_llm_7b modular_rag average +
python rag_eval.py deepseek_llm_7b modular_rag poor