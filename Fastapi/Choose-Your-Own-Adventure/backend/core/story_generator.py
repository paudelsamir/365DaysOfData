# here we'll create story generation logic, e.g. using ai models to generate story text

from sqlalchemy.orm import Session

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
import re

from models.story import Story, StoryNode
from core.models import StoryLLMResponse, StoryNodeLLM
from dotenv import load_dotenv
import os

load_dotenv()

class StoryGenerator:

    @classmethod
    def _get_llm(cls):
        # Reload environment variables to get latest values
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        # get Ollama configuration from environment or use defaults
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        
        # Check if we need to add version tag for Ollama
        if ":" not in ollama_model:
            # Try to find the model with version tag
            try:
                import requests
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name") for model in models]
                    
                    # Find model with version tag
                    for model_name in model_names:
                        if model_name.startswith(ollama_model + ":"):
                            ollama_model = model_name
                            break
            except Exception as e:
                print(f"Warning: Could not check available models: {e}")
        
        print(f"Initializing Ollama LLM with base_url: {ollama_base_url}, model: {ollama_model}")
        
        return OllamaLLM(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.7
        )

    @classmethod
    def generate_story(cls, db: Session, session_id: str, theme: str = "fantasy") -> Story:
        llm = cls._get_llm()

        # Simplified prompt to match our exact Pydantic model structure
        prompt_text = f"""Create a choose-your-own-adventure story about {theme}. 

Return ONLY valid JSON in this exact format:

{{
  "title": "Your Story Title",
  "rootNode": {{
    "content": "Story opening about {theme}",
    "isEnding": false,
    "isWinningEnding": false,
    "options": [
      {{
        "text": "First choice",
        "nextNode": {{
          "content": "Result of first choice",
          "isEnding": true,
          "isWinningEnding": true,
          "options": null
        }}
      }},
      {{
        "text": "Second choice",
        "nextNode": {{
          "content": "Result of second choice", 
          "isEnding": true,
          "isWinningEnding": false,
          "options": null
        }}
      }}
    ]
  }}
}}"""

        try:
            print(f"Attempting to generate story with Ollama for theme: {theme}")
            
            # Get Ollama configuration again for direct HTTP request
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
            
            # Use direct HTTP request to Ollama instead of LangChain
            import requests
            
            ollama_request = {
                "model": ollama_model,
                "prompt": prompt_text,
                "stream": False
            }
            
            response = requests.post(f"{ollama_base_url}/api/generate", 
                                   json=ollama_request, 
                                   timeout=30)
            response.raise_for_status()
            
            ollama_response = response.json()
            response_text = ollama_response.get("response", "")
            
            print(f"Raw Ollama response: {response_text}")  # Show full response
            
            # Clean up the response to extract JSON
            extracted_json = cls._extract_json(response_text)
            print(f"Extracted JSON: {extracted_json}")
            
            # Parse JSON and validate
            story_data = json.loads(extracted_json)
            print(f"Parsed story data: {story_data}")
            
            story_structure = StoryLLMResponse.model_validate(story_data)
            
            print("Successfully generated story with Ollama!")

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {str(e)}")
            print(f"Raw response that failed to parse: {locals().get('response_text', 'Not available')}")
            error_msg = f"JSON Error: {str(e)}"
            
            print("Creating fallback story...")
            fallback = cls._create_fallback_story(theme)
            fallback.title = f"Fallback: {theme} (JSON parse error)"
            story_structure = fallback
            
        except Exception as e:
            # More detailed error logging
            print(f"LLM generation failed with error: {type(e).__name__}: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response details: {e.response}")
            
            # Save the error to database for debugging
            error_msg = f"LLM Error: {type(e).__name__}: {str(e)}"
            
            # Check if Ollama is running
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    print("Ollama is running, but story generation failed")
                    error_msg += " | Ollama is running"
                else:
                    print(f"Ollama responded with status: {response.status_code}")
                    error_msg += f" | Ollama status: {response.status_code}"
            except requests.exceptions.ConnectionError:
                print("Ollama is not running or not accessible at http://localhost:11434")
                error_msg += " | Ollama not accessible"
            except Exception as conn_e:
                print(f"Error checking Ollama connection: {conn_e}")
                error_msg += f" | Connection error: {conn_e}"
            
            print("Creating fallback story...")
            
            # Create a story with error information in the title for debugging
            fallback = cls._create_fallback_story(theme)
            fallback.title = f"Fallback: {theme} ({error_msg[:50]}...)"
            story_structure = fallback

        # Save to database
        story_db = Story(title=story_structure.title, session_id=session_id)
        db.add(story_db)
        db.flush()

        root_node_data = story_structure.rootNode
        if isinstance(root_node_data, dict):
            root_node_data = StoryNodeLLM.model_validate(root_node_data)

        cls._process_story_node(db, story_db.id, root_node_data, is_root=True)

        db.commit()
        return story_db

    @classmethod
    def _extract_json(cls, text: str) -> str:
        """Extract JSON from text response"""
        text = text.strip()
        
        # First, try to extract content from markdown code blocks
        if "```" in text:
            # Find all code blocks
            code_blocks = []
            lines = text.split('\n')
            in_code_block = False
            current_block = []
            
            for line in lines:
                if line.strip().startswith('```'):
                    if in_code_block:
                        # End of code block
                        code_blocks.append('\n'.join(current_block))
                        current_block = []
                        in_code_block = False
                    else:
                        # Start of code block
                        in_code_block = True
                elif in_code_block:
                    current_block.append(line)
            
            # Try to find valid JSON in code blocks
            for block in code_blocks:
                block = block.strip()
                if block.startswith('{') and block.endswith('}'):
                    return block
        
        # If no code blocks, try to find JSON object directly
        brace_start = text.find('{')
        if brace_start != -1:
            brace_count = 0
            for i, char in enumerate(text[brace_start:], brace_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[brace_start:i+1]
                        # Quick validation that it looks like JSON
                        if candidate.count('{') == candidate.count('}') and candidate.count('"') >= 2:
                            return candidate
        
        return text

    @classmethod
    def _create_fallback_story(cls, theme: str) -> StoryLLMResponse:
        """Create a simple fallback story when LLM fails"""
        # Create the story data as dictionaries that can be properly validated
        story_data = {
            "title": f"The {theme.title()} Adventure",
            "rootNode": {
                "content": f"You find yourself at the beginning of a {theme} adventure. The path ahead splits in two directions.",
                "isEnding": False,
                "isWinningEnding": False,
                "options": [
                    {
                        "text": "Take the left path",
                        "nextNode": {
                            "content": "You chose the left path and discovered a treasure! You win!",
                            "isEnding": True,
                            "isWinningEnding": True,
                            "options": None
                        }
                    },
                    {
                        "text": "Take the right path", 
                        "nextNode": {
                            "content": "You chose the right path and fell into a trap. Game over!",
                            "isEnding": True,
                            "isWinningEnding": False,
                            "options": None
                        }
                    }
                ]
            }
        }
        
        return StoryLLMResponse.model_validate(story_data)

    @classmethod
    def _process_story_node(cls, db: Session, story_id: int, node_data: StoryNodeLLM, is_root: bool = False) -> StoryNode:
        node = StoryNode(
            story_id=story_id,
            content=node_data.content if hasattr(node_data, "content") else node_data["content"],
            is_root=is_root,
            is_ending=node_data.isEnding if hasattr(node_data, "isEnding") else node_data["isEnding"],
            is_winning_ending=node_data.isWinningEnding if hasattr(node_data, "isWinningEnding") else node_data["isWinningEnding"],
            options=[]
        )
        db.add(node)
        db.flush()

        if not node.is_ending and (hasattr(node_data, "options") and node_data.options):
            options_list = []
            for option_data in node_data.options:
                next_node = option_data.nextNode

                if isinstance(next_node, dict):
                    next_node = StoryNodeLLM.model_validate(next_node)

                child_node = cls._process_story_node(db, story_id, next_node, False)

                options_list.append({
                    "text": option_data.text,
                    "node_id": child_node.id
                })

            node.options = options_list

        db.flush()
        return node
