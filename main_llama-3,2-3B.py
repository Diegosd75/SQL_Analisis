import json
import logging
import argparse
from infrastructure.llama_evaluator import evaluate_query_llama
from domain.query_analyzer import analyze_query
from application.evaluate_query import execute_query_evaluation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_query_from_args():
    parser = argparse.ArgumentParser(description="Analyze SQL queries using Llama-3.2-3B.")
    parser.add_argument("--query", type=str, help="SQL query to analyze.")
    parser.add_argument("--file", type=str, help="Path to a file containing an SQL query.")
    args = parser.parse_args()

    if args.query:
        return args.query
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logging.error(f"Error reading file: {e}")
            return None
    else:
        logging.error("No SQL query provided. Use --query or --file.")
        return None

def main():
    query = get_query_from_args()
    if not query:
        return

    # Evaluate query using function-based approach
    result = execute_query_evaluation(query)

    # Print output as JSON
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
