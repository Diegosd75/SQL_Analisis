import torch
import torch.nn as nn
import os
from train import train_model

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class SimpleLlamaModel(nn.Module):
    def __init__(self):
        super(SimpleLlamaModel, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def load_model():
    print("Cargando modelo...")
    model = SimpleLlamaModel()
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
