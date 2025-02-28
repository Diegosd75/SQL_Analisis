import torch
from model import load_model
from features import extract_features

def classify_query(query, model):
    print("Clasificando el query en una escala de 1 a 5...")
    input_tensor = extract_features(query)
    output = model(input_tensor)
    score = torch.argmax(output, dim=1).item() + 1
    return score

def main():
    model = load_model()
    query = "SELECT id FROM users JOIN orders ON users.id = orders.user_id"
    quality = classify_query(query.strip(), model)
    print(f"Calidad del Query (1-5): {quality}")

if __name__ == "__main__":
    main()