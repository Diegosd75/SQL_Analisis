import torch
import torch.optim as optim
import torch.nn as nn
import json
from features import extract_features

def load_training_data():
    with open("sql_quality_dataset.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    
    all_problems = list({problem for item in data for problem in item["problems"]})
    problem_to_index = {problem: idx for idx, problem in enumerate(all_problems)}
    
    queries = []
    for item in data:
        query = item["query"]
        score = max(0, min(item["score"], 5)) 
        problem_vector = [0] * len(all_problems)
        for problem in item["problems"]:
            problem_vector[problem_to_index[problem]] = 1
        queries.append((query, score, problem_vector, item["solution"]))
    
    return queries, len(all_problems)  

def train_model(model):
    print("Entrenando el modelo...")
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    weights = torch.tensor([3.0, 2.0, 1.5, 1.0, 1.0, 0.5], dtype=torch.float32) 
    criterion_score = nn.CrossEntropyLoss(weight=weights)
    criterion_problems = nn.BCEWithLogitsLoss()
    
    queries, num_problems = load_training_data()
    
    for epoch in range(1000):
        total_loss = 0
        for query, score, problems, _ in queries:
            optimizer.zero_grad()
            input_tensor = extract_features(query)
            score_output, problems_output = model(input_tensor)
            loss_score = criterion_score(score_output, torch.tensor([score], dtype=torch.long))
            problem_tensor = torch.tensor([problems], dtype=torch.float32).view(1, -1)  
            loss_problems = criterion_problems(problems_output, problem_tensor)
            loss = loss_score + loss_problems
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    print("Entrenamiento finalizado.")
