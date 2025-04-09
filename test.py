import torch
from model import load_model
from features import extract_features

def test_queries():
    # Cargar el modelo entrenado
    model = load_model()

    # Lista de consultas SQL de prueba
    test_cases = [
        ("SELECT * FROM users;", "Consulta ineficiente con SELECT *"),
        ("INSERT INTO users (id, name) VALUES ();", "Falta de valores en INSERT"),
        ("DELETE users WHERE id=5;", "Error sintáctico: falta FROM en DELETE"),
        ("SELECT id FROM;", "Error sintáctico: falta tabla en FROM"),
        ("UPDATE users SET password='123';", "Error sintáctico: falta WHERE en UPDATE"),
        ("CREATE TABLE employees (id INT, name TEXT;", "Error sintáctico: falta paréntesis de cierre"),
        ("SELECT id FROM users WHERE email LIKE '%@gmail.com';", "Uso ineficiente de LIKE con '%' al inicio"),
        ("DROP TABLE users;", "Operación peligrosa: eliminación de tabla")
    ]

    print("\n Probando clasificación de consultas SQL...\n")
    
    for query, expected_issue in test_cases:
        input_tensor = extract_features(query)
        score_output, problems_output = model(input_tensor)

        # Convertir la salida del score en probabilidades
        score_probs = torch.nn.functional.softmax(score_output, dim=1)
        score = torch.argmax(score_probs, dim=1).item()

        # Extraer los 3 problemas más probables
        problems_indices = torch.topk(problems_output, 3).indices.tolist()[0]

        print(f"Query: {query}")
        print(f"Score Predicho: {score} (0 = peor, 4 = mejor)")
        print(f"Problema Esperado: {expected_issue}")
        print(f"Problemas Detectados (Índices): {problems_indices}")
        print("-" * 60)

if __name__ == "__main__":
    test_queries()
