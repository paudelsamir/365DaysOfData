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

        # Simple but clear prompt for branching story
        prompt_text = f"""Create a choose-your-own-adventure story about {theme}.

Make a story with these requirements:
1. Start with 4 choices
2. At least 2 choices should lead to MORE choices (not endings)
3. Each of those should have 4 more choices
4. Make it about {theme}

JSON format:
{{
  "title": "Story about {theme}",
  "rootNode": {{
    "content": "Story start (30 words max)",
    "isEnding": false,
    "isWinningEnding": false,
    "options": [
      {{
        "text": "Choice 1",
        "nextNode": {{
          "content": "What happens (30 words max)",
          "isEnding": false,
          "isWinningEnding": false,
          "options": [
            {{"text": "Sub choice 1A", "nextNode": {{"content": "Ending (30 words max)", "isEnding": true, "isWinningEnding": true, "options": null}}}},
            {{"text": "Sub choice 1B", "nextNode": {{"content": "Ending (30 words max)", "isEnding": true, "isWinningEnding": false, "options": null}}}},
            {{"text": "Sub choice 1C", "nextNode": {{"content": "Ending (30 words max)", "isEnding": true, "isWinningEnding": false, "options": null}}}},
            {{"text": "Sub choice 1D", "nextNode": {{"content": "Ending (30 words max)", "isEnding": true, "isWinningEnding": true, "options": null}}}}
          ]
        }}
      }},
      {{"text": "Choice 2", "nextNode": {{"content": "What happens", "isEnding": false, "isWinningEnding": false, "options": [{{"text": "Sub 2A", "nextNode": {{"content": "End", "isEnding": true, "isWinningEnding": true, "options": null}}}}, {{"text": "Sub 2B", "nextNode": {{"content": "End", "isEnding": true, "isWinningEnding": false, "options": null}}}}, {{"text": "Sub 2C", "nextNode": {{"content": "End", "isEnding": true, "isWinningEnding": false, "options": null}}}}, {{"text": "Sub 2D", "nextNode": {{"content": "End", "isEnding": true, "isWinningEnding": true, "options": null}}}}]}}}},
      {{"text": "Choice 3", "nextNode": {{"content": "Direct ending", "isEnding": true, "isWinningEnding": false, "options": null}}}},
      {{"text": "Choice 4", "nextNode": {{"content": "Direct ending", "isEnding": true, "isWinningEnding": true, "options": null}}}}
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
                                   timeout=120)
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
            
            # Check if the story has proper multi-level structure
            root_node = story_data.get('rootNode', {})
            options = root_node.get('options', [])
            
            # Count how many non-ending options we have (should be at least 2 for multi-level)
            non_ending_options = 0
            for option in options:
                next_node = option.get('nextNode', {})
                if not next_node.get('isEnding', True):
                    non_ending_options += 1
            
            # If we don't have enough non-ending paths, use fallback
            if non_ending_options < 2:
                print("Story lacks sufficient branching depth, using enhanced fallback...")
                story_structure = cls._create_fallback_story(theme)
            else:
                story_structure = StoryLLMResponse.model_validate(story_data)
                print("Successfully generated multi-level story with Ollama!")

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {str(e)}")
            print(f"Raw response that failed to parse: {locals().get('response_text', 'Not available')}")
            error_msg = f"JSON Error: {str(e)}"
            
            print("Creating fallback story...")
            fallback = cls._create_fallback_story(theme)
            # Keep the clean title from fallback story
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
            
            # Create clean fallback story without error message in title
            fallback = cls._create_fallback_story(theme)
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
        """Extract JSON from text response with better error handling"""
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
                    # Try to validate JSON structure
                    try:
                        import json
                        json.loads(block)  # Test if it's valid JSON
                        return block
                    except:
                        continue
        
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
                        # Try to validate this JSON
                        try:
                            import json
                            json.loads(candidate)  # Test if it's valid JSON
                            return candidate
                        except:
                            continue
        
        return text

    @classmethod
    def _create_fallback_story(cls, theme: str) -> StoryLLMResponse:
        """Create a deep multi-level fallback story with 4-5 levels when LLM fails"""
        # Create the story data as dictionaries that can be properly validated
        story_data = {
            "title": f"The {theme.title()} Adventure",
            "rootNode": {
                "content": f"You begin your {theme} adventure in a mysterious realm. Four paths stretch before you, each promising different destinies.",
                "isEnding": False,
                "isWinningEnding": False,
                "options": [
                    {
                        "text": "Take the northern path",
                        "nextNode": {
                            "content": "The northern path leads to an ancient temple with four glowing doorways. Each door hums with different energy.",
                            "isEnding": False,
                            "isWinningEnding": False,
                            "options": [
                                {
                                    "text": "Enter the golden door",
                                    "nextNode": {
                                        "content": "Inside, you find a wise oracle who presents you with four sacred trials to prove your worth.",
                                        "isEnding": False,
                                        "isWinningEnding": False,
                                        "options": [
                                            {
                                                "text": "Trial of courage",
                                                "nextNode": {
                                                    "content": "You face your deepest fears in a realm of shadows. Four choices determine your fate here.",
                                                    "isEnding": False,
                                                    "isWinningEnding": False,
                                                    "options": [
                                                        {"text": "Fight the shadow beast", "nextNode": {"content": "You defeat the beast and gain eternal courage!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                                        {"text": "Negotiate with shadows", "nextNode": {"content": "The shadows reject your words and consume you.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                                        {"text": "Use light magic", "nextNode": {"content": "Your light banishes all darkness and you become a beacon of hope!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                                        {"text": "Run away quickly", "nextNode": {"content": "You escape but lose your chance at greatness forever.", "isEnding": True, "isWinningEnding": False, "options": None}}
                                                    ]
                                                }
                                            },
                                            {"text": "Trial of wisdom", "nextNode": {"content": "You solve ancient riddles and become the wisest being in existence!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                            {"text": "Trial of strength", "nextNode": {"content": "You lift impossible weights but your body breaks under the strain.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                            {"text": "Trial of heart", "nextNode": {"content": "You show pure compassion and become a beloved guardian spirit!", "isEnding": True, "isWinningEnding": True, "options": None}}
                                        ]
                                    }
                                },
                                {
                                    "text": "Enter the silver door",
                                    "nextNode": {
                                        "content": "You enter a realm of time where past, present, and future swirl around you in chaos.",
                                        "isEnding": False,
                                        "isWinningEnding": False,
                                        "options": [
                                            {
                                                "text": "Travel to the past",
                                                "nextNode": {
                                                    "content": "In the past, you discover four ancient kingdoms at war. Your choice here changes history itself.",
                                                    "isEnding": False,
                                                    "isWinningEnding": False,
                                                    "options": [
                                                        {"text": "Unite the kingdoms", "nextNode": {"content": "You become the legendary peacemaker who ended the eternal wars!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                                        {"text": "Join the strongest army", "nextNode": {"content": "You help create a tyranny that oppresses the realm for centuries.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                                        {"text": "Stay neutral", "nextNode": {"content": "Your neutrality allows the wars to continue indefinitely.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                                        {"text": "Steal their secrets", "nextNode": {"content": "You gather ancient knowledge and become a master of lost arts!", "isEnding": True, "isWinningEnding": True, "options": None}}
                                                    ]
                                                }
                                            },
                                            {"text": "Travel to the future", "nextNode": {"content": "You see a dystopian world and realize your choices led to this outcome.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                            {"text": "Stay in the present", "nextNode": {"content": "You master the flow of time and become a temporal guardian!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                            {"text": "Try to escape", "nextNode": {"content": "You become lost between time streams and exist nowhere forever.", "isEnding": True, "isWinningEnding": False, "options": None}}
                                        ]
                                    }
                                },
                                {"text": "Enter the crystal door", "nextNode": {"content": "You find a mirror that reflects your true nature and achieve enlightenment!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                {"text": "Enter the obsidian door", "nextNode": {"content": "Dark forces trap you in eternal servitude to shadow masters.", "isEnding": True, "isWinningEnding": False, "options": None}}
                            ]
                        }
                    },
                    {
                        "text": "Take the eastern path",
                        "nextNode": {
                            "content": "The eastern path leads to a magical forest where four elemental spirits await your decision.",
                            "isEnding": False,
                            "isWinningEnding": False,
                            "options": [
                                {
                                    "text": "Commune with fire spirit",
                                    "nextNode": {
                                        "content": "The fire spirit offers to teach you the ways of flame. Four lessons await you.",
                                        "isEnding": False,
                                        "isWinningEnding": False,
                                        "options": [
                                            {"text": "Learn to create fire", "nextNode": {"content": "You master fire creation and become a legendary fire mage!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                            {"text": "Learn to control fire", "nextNode": {"content": "You gain fire immunity but lose the ability to feel warmth forever.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                            {"text": "Learn to become fire", "nextNode": {"content": "You transform into pure flame and protect the forest eternally!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                            {"text": "Reject the fire spirit", "nextNode": {"content": "The spirit burns you to ash for your disrespect.", "isEnding": True, "isWinningEnding": False, "options": None}}
                                        ]
                                    }
                                },
                                {"text": "Commune with water spirit", "nextNode": {"content": "You learn water magic but your body becomes liquid forever.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                {"text": "Commune with earth spirit", "nextNode": {"content": "You gain the strength of mountains and become an earth guardian!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                {"text": "Commune with air spirit", "nextNode": {"content": "You learn to fly but lose your connection to the ground forever.", "isEnding": True, "isWinningEnding": False, "options": None}}
                            ]
                        }
                    },
                    {
                        "text": "Take the southern path",
                        "nextNode": {
                            "content": "The southern path leads to a crossroads where four merchants offer you mysterious bargains.",
                            "isEnding": False,
                            "isWinningEnding": False,
                            "options": [
                                {"text": "Trade with gold merchant", "nextNode": {"content": "You gain infinite wealth but lose your soul to greed.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                {"text": "Trade with knowledge merchant", "nextNode": {"content": "You gain all knowledge but your mind cannot handle the truth.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                {"text": "Trade with power merchant", "nextNode": {"content": "You gain ultimate power and use it to protect the innocent!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                {"text": "Refuse all trades", "nextNode": {"content": "You keep your humanity and find happiness in simple things!", "isEnding": True, "isWinningEnding": True, "options": None}}
                            ]
                        }
                    },
                    {
                        "text": "Take the western path",
                        "nextNode": {
                            "content": "The western path leads to an ancient library with four mystical tomes waiting to be opened.",
                            "isEnding": False,
                            "isWinningEnding": False,
                            "options": [
                                {
                                    "text": "Read the tome of destiny",
                                    "nextNode": {
                                        "content": "The tome shows you four possible destinies. You must choose which future to make real.",
                                        "isEnding": False,
                                        "isWinningEnding": False,
                                        "options": [
                                            {"text": "Hero's destiny", "nextNode": {"content": "You become the greatest hero of all time, saving countless lives!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                            {"text": "Ruler's destiny", "nextNode": {"content": "You become a just ruler but the burden of leadership crushes your spirit.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                            {"text": "Scholar's destiny", "nextNode": {"content": "You discover the secrets of the universe and transcend mortality!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                            {"text": "Wanderer's destiny", "nextNode": {"content": "You travel forever but never find a place to call home.", "isEnding": True, "isWinningEnding": False, "options": None}}
                                        ]
                                    }
                                },
                                {"text": "Read the tome of secrets", "nextNode": {"content": "You learn forbidden knowledge and become mad with power.", "isEnding": True, "isWinningEnding": False, "options": None}},
                                {"text": "Read the tome of wisdom", "nextNode": {"content": "You gain perfect wisdom and guide others to enlightenment!", "isEnding": True, "isWinningEnding": True, "options": None}},
                                {"text": "Read the tome of nightmares", "nextNode": {"content": "The tome traps you in eternal nightmares from which you never wake.", "isEnding": True, "isWinningEnding": False, "options": None}}
                            ]
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
