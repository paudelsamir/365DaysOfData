# usually contains fastapi app, includes routers, starts uvicorn server etc


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.configs import settings
from routers import story, job
from db.database import create_tables

app = FastAPI(
    title = "Choose Your Own Adventure",
    description = "api to generate cool stories",
    version = "0.1.0",
    docs_url = "/docs",
    redoc_url = "/redoc"
)

# Create database tables
create_tables()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(story.router, prefix=settings.API_PREFIX)
app.include_router(job.router, prefix=settings.API_PREFIX)

@app.get("/")
async def root():
    return {
        "message": "Choose Your Own Adventure API is running!",
        "version": "0.1.0",
        "endpoints": {
            "docs": "/docs",
            "stories": "/api/stories",
            "jobs": "/api/jobs"
        }
    }

@app.get("/debug/ollama-test")
async def test_ollama_generation():
    """Test Ollama generation with a simple prompt for debugging"""
    import requests
    from dotenv import load_dotenv
    import os
    
    load_dotenv(override=True)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    
    test_prompt = """Create a simple JSON object: {"title": "Test Story", "content": "This is a test"}"""
    
    try:
        ollama_request = {
            "model": ollama_model,
            "prompt": test_prompt,
            "stream": False
        }
        
        response = requests.post(f"{ollama_base_url}/api/generate", 
                               json=ollama_request, 
                               timeout=30)
        response.raise_for_status()
        
        ollama_response = response.json()
        raw_text = ollama_response.get("response", "")
        
        return {
            "ollama_request": ollama_request,
            "ollama_full_response": ollama_response,
            "raw_response_text": raw_text,
            "text_length": len(raw_text)
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health/ollama")
async def check_ollama():
    """Check if Ollama is running and accessible"""
    import requests
    from dotenv import load_dotenv
    import os
    
    # Reload environment variables
    load_dotenv(override=True)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    
    try:
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            # Check if target model is available (with or without version tag)
            model_available = any(
                model_name.startswith(ollama_model) or 
                model_name == ollama_model 
                for model_name in model_names
            )
            
            return {
                "status": "connected",
                "ollama_url": ollama_base_url,
                "target_model": ollama_model,
                "available_models": model_names,
                "model_available": model_available
            }
        else:
            return {
                "status": "error", 
                "message": f"Ollama responded with status {response.status_code}"
            }
    except requests.exceptions.ConnectionError:
        return {
            "status": "disconnected",
            "message": f"Cannot connect to Ollama at {ollama_base_url}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app = "main:app", host = "0.0.0.0", port = 8000, reload = True)