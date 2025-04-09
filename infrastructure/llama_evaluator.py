import json
import re
from transformers import pipeline

def extract_json(text):
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_text = json_match.group(0).strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass
    return None

def load_llama_model():
    """Loads the Llama-3.2-3B model."""
    return pipeline(
        "text-generation",
        model="./Llama-3.2-3B",
        return_full_text=False,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=150
    )

def evaluate_query_llama(query: str, model) -> dict:
    """Evaluates an SQL query using Llama-3.2-3B."""
    
    print(f"\n🧐 Evaluating Query:\n{query}\n")

    prompt = (
        f"Analyze the following SQL query and respond in JSON format:\n\n"
        f"SQL Query:\n{query}\n\n"
        f"Response Format:\n"
        f"{{\n"
        f"    \"score\": <integer from 0 to 5>,\n"
        f"    \"issues\": [\"list of detected issues\"],\n"
        f"    \"suggestions\": [\"list of improvement suggestions\"]\n"
        f"}}\n\n"
        f"Provide your analysis below:\n"
    )

    for attempt in range(2):
        response = model(prompt, max_new_tokens=150, do_sample=False, temperature=0.0, top_p=1.0, truncation=True)
        result = response[0]['generated_text'].strip()

        print("\n🔎 Raw Model Output:\n", result)

        parsed_json = extract_json(result)
        if parsed_json:
            return parsed_json
        else:
            print(f"⚠️ Attempt {attempt+1}: Model returned invalid JSON. Retrying...")

    return {
        "score": 0,
        "issues": ["The model failed to analyze the SQL query."],
        "suggestions": ["Try adjusting the query or reloading the model."]
    }
