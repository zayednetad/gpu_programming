# Step 1: Install required packages
!pip install -q transformers accelerate fastapi uvicorn nest_asyncio pyngrok huggingface_hub
# Step 2: Insert this before loading the model
from huggingface_hub import login
login("your hugging face token")
# Step 3: Load Mistral model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

from pyngrok import conf
conf.get_default().auth_token = "your ngrok token"

# Step 4: FastAPI setup
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from threading import Thread

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Mistral Chat</title></head>
    <body>
      <h2>💬 Chat with Mistral</h2>
      <textarea id='prompt' rows='5' cols='80'></textarea><br>
      <button onclick='ask()'>Send</button>
      <pre id='response' style='white-space:pre-wrap;background:#f0f0f0;padding:1em;'></pre>
      <script>
        async function ask() {
          const prompt = document.getElementById("prompt").value;
          const res = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({prompt})
          });
          const data = await res.json();
          document.getElementById("response").textContent = data.response;
        }
      </script>
    </body>
    </html>
    """

class Prompt(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: Prompt):
    output = generator(req.prompt, max_new_tokens=200)[0]["generated_text"]
    return {"response": output}

# Step 5: Launch FastAPI + ngrok
public_url = ngrok.connect(8000)
print(f"🔗 Public Chat URL: {public_url}")

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

nest_asyncio.apply()
Thread(target=run).start()
