import requests

try:
    r = requests.get("http://localhost:11434/api/tags")
    if r.status_code == 200:
        print("✅ Connected to Ollama!")
        print("Available models:", [m['name'] for m in r.json().get('models', [])])
    else:
        print("⚠️ Ollama responded with:", r.status_code, r.text)
except Exception as e:
    print("❌ Could not connect:", e)
