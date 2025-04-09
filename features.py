import torch
import re

def check_parentheses_balance(query):
    """ Verifica si los paréntesis están balanceados """
    return query.count('(') == query.count(')')

def extract_features(query):
    
    #Errores de sintaxis
    syntax_errors = [
        int(bool(re.search(r"INSERT INTO \w+ \(.+\) VALUES \(\);", query, re.IGNORECASE))),  
        int(bool(re.search(r"DELETE \w+ WHERE", query, re.IGNORECASE))),  
        int(bool(re.search(r"SELECT \w+ FROM;", query, re.IGNORECASE))),  
        int(bool(re.search(r"UPDATE \w+ SET .+ WHERE;", query, re.IGNORECASE))),  
        int(bool(re.search(r"CREATE TABLE \w+ \(.+;", query, re.IGNORECASE))),  
        int(not check_parentheses_balance(query))  
    ]
    
    #Caracteristicas generales del query
    general_features = [
        int(bool(re.search(r"SELECT \*", query, re.IGNORECASE))),
        len(re.findall(r"JOIN", query, re.IGNORECASE)),
        len(re.findall(r"\(SELECT", query, re.IGNORECASE)),
        len(re.findall(r"WHERE", query, re.IGNORECASE)),
        int(bool(re.search(r"INDEX", query, re.IGNORECASE))),
        int(bool(re.search(r"LIMIT", query, re.IGNORECASE))),
        len(re.findall(r"UPDATE", query, re.IGNORECASE)),
        len(re.findall(r"INSERT INTO", query, re.IGNORECASE)),
        int(bool(re.search(r"LIKE '%", query, re.IGNORECASE))),
        int(bool(re.search(r" OR 1=1", query, re.IGNORECASE))),
        int(bool(re.search(r"EXISTS", query, re.IGNORECASE))),
        len(re.findall(r"GROUP BY", query, re.IGNORECASE)),
        int(bool(re.search(r"YEAR\(|MONTH\(", query, re.IGNORECASE))),
        int(bool(re.search(r"ORDER BY", query, re.IGNORECASE)))
    ]

    return torch.tensor(general_features + syntax_errors, dtype=torch.float32).unsqueeze(0)
