import torch
import torch.nn.functional as F
import json
from model import load_model
from features import extract_features

# Cargar el diccionario de problemas y soluciones desde el dataset
def load_problem_mapping():
    with open("sql_quality_dataset.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    all_problems = list({problem for item in data for problem in item["problems"]})
    problem_to_solution = {problem: item["solution"] for item in data for problem in item["problems"]}
    return {idx: (problem, problem_to_solution.get(problem, "No hay solución disponible")) for idx, problem in enumerate(all_problems)}

# Clasificar la consulta SQL
def classify_query(query, model, problem_mapping):
    input_tensor = extract_features(query)
    score_output, problems_output = model(input_tensor) 
    score_probs = F.softmax(score_output, dim=1)
    score = torch.argmax(score_probs, dim=1).item()  
    problems_indices = torch.topk(problems_output, 3).indices.tolist()[0]
    problems_solutions = [problem_mapping.get(idx, ("Problema desconocido", "No hay solución disponible")) for idx in problems_indices]
    
    return score, problems_solutions

def main():
    model = load_model()
    problem_mapping = load_problem_mapping()
    query = "SELECT o.order_id, c.name, o.total FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = 'completed' ORDER BY o.order_date DESC LIMIT 10;"
    score, problems_solutions = classify_query(query.strip(), model, problem_mapping)
    
    print(f"Calidad del Query (1-5): {score}")
    print("Posibles Problemas y Soluciones:")
    if score==5:
        problem, solution = problems_solutions[0]  # Obtener el primer problema y solución
        print(f"- No significant issues; query is well-optimized\n  Solución: Query is well-structured and optimized. Ensure indexes exist on customer_id and status for optimal performance.\n")
        
    else:
        for problem, solution in problems_solutions:
            print(f"- {problem}\n  Solución: {solution}\n")
            

if __name__ == "__main__":
    main()
