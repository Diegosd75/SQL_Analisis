import json
import re
from transformers import pipeline

def load_llama_model():
    """Loads the Llama-3.2-3B model for chatbot interaction."""
    return pipeline(
        "text-generation",
        model="./Llama-3.2-3B",
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=20
    )

def chat_with_model(model):
    """Interactive chatbot loop to communicate with Llama-3.2-3B."""
    print("\n🤖 Chatbot Mode Activated. Type 'exit' to quit. 🤖\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! 👋")
            break
        response = model(user_input, max_new_tokens=100, truncation=False)
        result = response[0]['generated_text'].strip()
        print("Chatbot:", result, "\n")

if __name__ == "__main__":
    model = load_llama_model()
    chat_with_model(model)
