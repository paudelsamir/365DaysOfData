import os
import textworld
from textworld import EnvInfos

# make a game first (only once, can reuse the .ulx file)
!tw-make custom --world-size 5 --quest-length 6 --nb-objects 10 --output tw_games/game.ulx -f -v --seed 456

# ask for room description, inventory, feedback
infos = EnvInfos(
    feedback=True,
    description=True,
    inventory=True,
    objective=True
)

# start the environment
TW_ENV = textworld.start("tw_games/game.ulx", infos)


# ------------------ helper classes ------------------

class HiddenPrints:
    """Context manager to suppress stdout temporarily."""
    def __enter__(self):
        import sys
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    def __exit__(self, *args):
        import sys
        sys.stdout.close()
        sys.stdout = self._original_stdout


class TextWorldInteract:
    """Wrapper around a TextWorld environment for tool use."""

    def __init__(self, textworld_env):
        self.textworld_env = textworld_env

    def play(self, command: str) -> str:
        if self.textworld_env is None:
            raise ValueError("TextWorld environment not set")
        with HiddenPrints():
            game_state, reward, done = self.textworld_env.step(command)
        return game_state["feedback"]


# ------------------ langchain agent ------------------

from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# initialize the tool
textworld_tool = TextWorldInteract(TW_ENV)

tools = [
    Tool(
        name="Play",
        func=textworld_tool.play,
        description="use this to interact with the TextWorld game environment"
    )
]

# use ollama llama3.2 as LLM
llm = OllamaLLM(model="llama3.2", temperature=0)

# prompt for ReAct style agent
prompt = PromptTemplate.from_template(
    """You are playing a TextWorld text adventure game.
You can only act by issuing valid game commands (like: look, inventory, go north, open door, take key).
Always use the Play tool to send commands and observe feedback.

Game state:
{input}

Think step by step. 
{agent_scratchpad}"""
)

# create agent and executor
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


game_state = TW_ENV.reset()
init_text = game_state.objective + "\n\n" + game_state.description

result = executor.invoke({"input": init_text})
print("FINAL RESULT:", result)
