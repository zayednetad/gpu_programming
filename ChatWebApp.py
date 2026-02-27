# ============================================================================
# Mistral Chat WebApp
# ============================================================================
# Installation: pip install transformers accelerate fastapi uvicorn nest_asyncio 
#               pyngrok huggingface_hub python-dotenv slowapi

import os
import logging
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from pyngrok import conf, ngrok
import nest_asyncio
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
import time
from threading import Thread

# ============================================================================
# Configuration & Logging
# ============================================================================
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
NGROK_TOKEN = os.getenv("NGROK_TOKEN", "").strip()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200"))
MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")
PORT = int(os.getenv("PORT", "8000"))
USE_NGROK = os.getenv("USE_NGROK", "true").lower() == "true"

if not HF_TOKEN:
    logger.warning(" HF_TOKEN not set in .env file. Set it to use Hugging Face models.")
if not NGROK_TOKEN and USE_NGROK:
    logger.warning(" NGROK_TOKEN not set in .env file. Public URL won't be available.")

# ============================================================================
# Initialize Model
# ============================================================================
def load_model():
    """Load Mistral model with proper error handling"""
    try:
        logger.info(f" Loading model: {MODEL_ID}")
        
        if HF_TOKEN:
            login(HF_TOKEN)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
        
        logger.info(" Model loaded successfully")
        return generator, tokenizer
    except Exception as e:
        logger.error(f" Failed to load model: {str(e)}")
        raise

try:
    generator, tokenizer = load_model()
except Exception as e:
    logger.error(f"Critical error during initialization: {str(e)}")
    raise

# ============================================================================
# FastAPI Setup
# ============================================================================
app = FastAPI(
    title="Mistral Chat API",
    description="Advanced chat interface powered by Mistral-7B",
    version="2.0"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Request/Response Models
# ============================================================================
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="User prompt")
    max_tokens: int = Field(default=MAX_TOKENS, ge=10, le=500, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Temperature for sampling")

class ChatResponse(BaseModel):
    response: str
    tokens_used: Optional[int] = None
    generation_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str

# ============================================================================
# Helper Functions
# ============================================================================
def count_tokens(text: str) -> int:
    """Count tokens in text"""
    try:
        return len(tokenizer.encode(text))
    except:
        return len(text.split())

def generate_response(prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = 0.7) -> tuple:
    """Generate response with error handling and timing"""
    start_time = time.time()
    
    try:
        # Validate input
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Invalid prompt")
        
        prompt = prompt.strip()
        if len(prompt) == 0:
            raise ValueError("Prompt cannot be empty")
        
        input_tokens = count_tokens(prompt)
        logger.info(f"Generating response for prompt ({input_tokens} tokens)...")
        
        # Generate with timeout handling
        outputs = generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            return_full_text=False
        )
        
        response_text = outputs[0]["generated_text"].strip()
        generation_time = time.time() - start_time
        output_tokens = count_tokens(response_text)
        
        logger.info(f" Response generated in {generation_time:.2f}s ({output_tokens} tokens)")
        
        return response_text, generation_time, output_tokens
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f" Generation error: {str(e)}")
        raise

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title> Mistral Chat</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 800px;
                width: 100%;
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px 20px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2em;
                margin-bottom: 5px;
            }
            
            .header p {
                opacity: 0.9;
                font-size: 0.95em;
            }
            
            .content {
                padding: 30px;
            }
            
            .input-section {
                margin-bottom: 20px;
            }
            
            .label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
            }
            
            textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                font-family: 'Courier New', monospace;
                resize: vertical;
                min-height: 100px;
                transition: border-color 0.3s;
            }
            
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .controls {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .control-group {
                display: flex;
                flex-direction: column;
            }
            
            .control-group input {
                padding: 10px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            
            .control-group input:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .button-group {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            
            button {
                flex: 1;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            .btn-send {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .btn-send:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            
            .btn-send:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .btn-clear {
                background: #f0f0f0;
                color: #333;
            }
            
            .btn-clear:hover {
                background: #e0e0e0;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #667eea;
                font-weight: 600;
            }
            
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .response-section {
                margin-top: 20px;
            }
            
            .response {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 15px;
                border-radius: 8px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.6;
                max-height: 400px;
                overflow-y: auto;
            }
            
            .stats {
                display: flex;
                gap: 20px;
                margin-top: 10px;
                font-size: 12px;
                color: #666;
            }
            
            .stat {
                display: flex;
                align-items: center;
            }
            
            .stat-icon {
                margin-right: 5px;
            }
            
            .error {
                background: #fee;
                border-left: 4px solid #c33;
                padding: 15px;
                border-radius: 8px;
                color: #c33;
                margin-top: 20px;
            }
            
            .success {
                background: #efe;
                border-left: 4px solid #3c3;
                padding: 15px;
                border-radius: 8px;
                color: #3c3;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1> Mistral Chat</h1>
                <p>Powered by Mistral-7B-Instruct</p>
            </div>
            
            <div class="content">
                <div class="input-section">
                    <label class="label">Your Message</label>
                    <textarea id="prompt" placeholder="Ask me anything..."></textarea>
                </div>
                
                <div class="controls">
                    <div class="control-group">
                        <label class="label">Max Tokens</label>
                        <input type="number" id="maxTokens" value="200" min="10" max="500">
                    </div>
                    <div class="control-group">
                        <label class="label">Temperature (0.1-2.0)</label>
                        <input type="number" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1">
                    </div>
                </div>
                
                <div class="button-group">
                    <button class="btn-send" id="sendBtn" onclick="ask()"> Send</button>
                    <button class="btn-clear" onclick="clearChat()"> Clear</button>
                </div>
                
                <div class="loading" id="loading">
                    <span class="spinner"></span>
                    Generating response...
                </div>
                
                <div class="response-section">
                    <div id="message"></div>
                </div>
            </div>
        </div>
        
        <script>
            let isLoading = false;
            
            async function ask() {
                const prompt = document.getElementById("prompt").value.trim();
                const maxTokens = parseInt(document.getElementById("maxTokens").value);
                const temperature = parseFloat(document.getElementById("temperature").value);
                const sendBtn = document.getElementById("sendBtn");
                const messageDiv = document.getElementById("message");
                const loading = document.getElementById("loading");
                
                // Validation
                if (!prompt) {
                    messageDiv.innerHTML = '<div class="error"> Please enter a message</div>';
                    return;
                }
                
                if (isLoading) {
                    messageDiv.innerHTML = '<div class="error"> Already generating a response...</div>';
                    return;
                }
                
                isLoading = true;
                sendBtn.disabled = true;
                loading.style.display = "block";
                messageDiv.innerHTML = "";
                
                try {
                    const response = await fetch("/chat", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({
                            prompt: prompt,
                            max_tokens: maxTokens,
                            temperature: temperature
                        })
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || "API error");
                    }
                    
                    const data = await response.json();
                    
                    let statsHtml = `
                        <div class="stats">
                            <div class="stat"><span class="stat-icon"></span>${data.generation_time.toFixed(2)}s</div>
                            <div class="stat"><span class="stat-icon"></span>${data.tokens_used} tokens</div>
                            <div class="stat"><span class="stat-icon"></span>${data.timestamp}</div>
                        </div>
                    `;
                    
                    messageDiv.innerHTML = `
                        <div class="response">${escapeHtml(data.response)}</div>
                        ${statsHtml}
                    `;
                    
                } catch (error) {
                    messageDiv.innerHTML = `<div class="error"> Error: ${error.message}</div>`;
                } finally {
                    isLoading = false;
                    sendBtn.disabled = false;
                    loading.style.display = "none";
                }
            }
            
            function clearChat() {
                document.getElementById("prompt").value = "";
                document.getElementById("message").innerHTML = "";
                document.getElementById("prompt").focus();
            }
            
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            // Allow Enter key to send message
            document.getElementById("prompt").addEventListener("keypress", function(e) {
                if (e.key === "Enter" && e.ctrlKey) {
                    ask();
                }
            });
            
            // Focus on load
            window.addEventListener("load", function() {
                document.getElementById("prompt").focus();
            });
        </script>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    device = "GPU" if torch.cuda.is_available() else "CPU"
    return HealthResponse(
        status="healthy",
        model=MODEL_ID,
        device=device
    )

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest):
    """Chat endpoint with rate limiting"""
    try:
        if not req.prompt or not req.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        response_text, generation_time, tokens = generate_response(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature
        )
        
        return ChatResponse(
            response=response_text,
            tokens_used=tokens,
            generation_time=generation_time,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch")
async def batch_chat(requests: list[ChatRequest]):
    """Batch chat endpoint for multiple requests"""
    try:
        results = []
        for req in requests:
            response_text, generation_time, tokens = generate_response(
                req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature
            )
            results.append({
                "prompt": req.prompt,
                "response": response_text,
                "tokens_used": tokens,
                "generation_time": generation_time
            })
        return {"results": results}
    except Exception as e:
        logger.error(f"Batch error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Server Launch
# ============================================================================
def run_server():
    """Run the uvicorn server"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )

if __name__ == "__main__":
    try:
        # Setup ngrok if enabled
        if USE_NGROK and NGROK_TOKEN:
            conf.get_default().auth_token = NGROK_TOKEN
            public_url = ngrok.connect(PORT)
            logger.info(f" Public URL: {public_url}")
        
        logger.info(f" Starting server on http://localhost:{PORT}")
        logger.info(" Open the URL above in your browser to access the chat")
        
        # Apply asyncio fix for nested event loops
        nest_asyncio.apply()
        
        # Run server
        run_server()
        
    except KeyboardInterrupt:
        logger.info("\n Server stopped by user")
    except Exception as e:
        logger.error(f" Server error: {str(e)}")
        raise
