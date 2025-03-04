from ollama import Client
from prompts import CORRECT, PARTIALLY, CONTRADICTORY, IRRELEVANT, NON_DOMAIN

HOST = 'http://catalpa-llm.fernuni-hagen.de:11434'
# HOST = 'localhost:11434'

def invoke_llm(prompt):
    client = Client(host=HOST)
    
    response = client.chat(model="deepseek-v2:latest", messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    return response['message']['content']

def print_answers(prompt_label, question, num_runs=6):
    print(prompt_label)
    print("CORRECT")
    for i in range(num_runs):
        print(invoke_llm(CORRECT.format(question=question)))
    print("\n")
    print("PARTIALLY")
    for i in range(num_runs):
        print(invoke_llm(PARTIALLY.format(question=question)))
    print("\n")
    print("CONTRADICTORY")
    for i in range(num_runs):
        print(invoke_llm(CONTRADICTORY.format(question=question)))
    print("\n")
    print("IRRELEVANT")
    for i in range(num_runs):
        print(invoke_llm(IRRELEVANT.format(question=question)))
    print("\n")
    print("NON_DOMAIN")
    for i in range(num_runs):
        print(invoke_llm(NON_DOMAIN.format(question=question)))
    print("\n")

VB1 = "How do you define a controlled experiment?"
PS4bp = """
Darla tied one end of a string around a doorknob and held the other end in her hand. 
When she plucked the string (pulled and let go quickly) she heard a sound. 
How would the pitch change if Darla made the string longer?
"""
ME27b = "How can you use a magnet to find out if the key is iron or aluminum?"

print_answers("VB1", VB1)
print_answers("PS4bp", PS4bp)
print_answers("ME27b", ME27b)
