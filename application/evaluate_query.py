from infrastructure.llama_evaluator import evaluate_query_llama, load_llama_model
from domain.query_analyzer import analyze_query

def execute_query_evaluation(query: str) -> dict:
    """Executes query evaluation by combining Llama-3.2-3B analysis with static SQL analysis."""
    model = load_llama_model()  # Load Llama model once

    model_result = evaluate_query_llama(query, model)
    analysis_result = analyze_query(query)

    return {
        "score": min(model_result["score"], analysis_result["score"]),
        "issues": model_result["issues"] + analysis_result["issues"],
        "suggestions": model_result["suggestions"] + analysis_result["suggestions"]
    }
