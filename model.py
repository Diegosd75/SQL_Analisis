import torch
import torch.nn as nn
import os
from train import train_model, load_training_data

#Reproducibilidad de los datos y usar algoritmos deterministicos
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

#se crea la clase modelo
class SQLQualityModel(nn.Module):
    def __init__(self, num_problems):
        super(SQLQualityModel, self).__init__()
        self.fc1 = nn.Linear(20, 32) 
        self.fc2 = nn.Linear(32, 16)
        self.fc3_score = nn.Linear(16, 6)  
        self.fc3_problems = nn.Linear(16, num_problems)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) 
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.relu(self.fc2(x))
        score_output = self.fc3_score(x)
        problems_output = self.fc3_problems(x)
        return score_output, problems_output

def load_model():
    print("Cargando modelo...")
    _, num_problems = load_training_data()
    model = SQLQualityModel(num_problems)
    model_path = "query_classifier.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Pesos del modelo cargados correctamente.")
    else:
        print("No se encontraron pesos guardados, entrenando modelo...")
        train_model(model)
        torch.save(model.state_dict(), model_path)
        print("Pesos del modelo guardados en", model_path)
    model.eval()
    return model
