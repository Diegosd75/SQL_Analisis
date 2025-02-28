import torch
import torch.optim as optim
import torch.nn as nn
from features import extract_features
from query_training_data import get_training_data

def train_model(model):
    print("Entrenando el modelo con 100 ejemplos...")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    queries = get_training_data()
    
    for epoch in range(200):
        total_loss = 0
        for query, label in queries:
            optimizer.zero_grad()
            input_tensor = extract_features(query)
            output = model(input_tensor)
            loss = criterion(output, torch.tensor([label - 1], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    print("Entrenamiento finalizado.")
