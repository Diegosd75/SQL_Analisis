import torch
import re

def extract_features(query):
    features = [
        int(bool(re.search(r"SELECT \*", query, re.IGNORECASE))),
        len(re.findall(r"JOIN", query, re.IGNORECASE)),
        len(re.findall(r"\(SELECT", query, re.IGNORECASE)),
        len(re.findall(r"WHERE", query, re.IGNORECASE)),
        int(bool(re.search(r"INDEX", query, re.IGNORECASE))),
        int(bool(re.search(r"LIMIT", query, re.IGNORECASE))),
        len(re.findall(r"UPDATE", query, re.IGNORECASE)),
        len(re.findall(r"INSERT INTO", query, re.IGNORECASE))
    ]
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)