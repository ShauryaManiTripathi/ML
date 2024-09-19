import requests
import json

OLLAMA_URL = "http://172.16.2.17:11434"

def get_ollama_info():
    url = f"{OLLAMA_URL}/api/show"
    data = {"name": "gemma:2b"}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        info = json.loads(response.text)
        return info.get('details', {}).get('hardware', 'Unknown')
    else:
        return f"Error: {response.status_code}"

def send_message(message):
    url = f"{OLLAMA_URL}/api/generate"
    data = {
        "model": "gemma:2b",
        "prompt": message,
        "stream": False
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return json.loads(response.text)['response']
    else:
        return f"Error: {response.status_code}"

def main():
    try:
        hardware = get_ollama_info()
        print(f"Ollama is running on: {hardware}")
        print("Welcome to the Ollama Chat with Gemma 2B! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            response = send_message(user_input)
            print("Gemma:", response)
    except requests.exceptions.ConnectionError:
        print(f"Error: Unable to connect to Ollama at {OLLAMA_URL}")
        print("Please check if the IP and port are correct and if Ollama is running.")

if __name__ == "__main__":
    main()