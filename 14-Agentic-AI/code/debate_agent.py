from typing import TypedDict
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

END = "END"


# --- debate state ---
class DebateState(TypedDict):
    messages: list          # full transcript
    next: str               # which agent speaks next
    round: int              # current round
    max_rounds: int         # stop after N rounds
    topic: str              # the debate topic
    sides: dict             # which side each agent defends
    scores: dict            # track scores per agent
    highlights: dict        # best one-liner or argument from each

# --- workflow engine ---
class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.entry_point = None
    
    def add_node(self, name, func):
        self.nodes[name] = func
    
    def set_entry_point(self, name):
        self.entry_point = name
    
    def get_node(self, name):
        return self.nodes.get(name)

def science_agent(state: DebateState):
    """Agent that argues based on science, logic, and evidence."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a science debater. Defend the side: {state['sides']['science']}. "
                   "Respond with one short, sharp paragraph using scientific reasoning and facts. "
                   "Be persuasive but concise."),
        ("user", "{messages}")
    ])
    chain = prompt | llm
    response = chain.invoke({"messages": state["messages"]})
    
    # log message
    state["messages"].append(f"Science: {response}")
    
    # update next turn
    state["next"] = "religion"
    return state


def religion_agent(state: DebateState):
    """Agent that argues from religious/God perspective."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a religious debater. Defend the side: {state['sides']['religion']}. "
                   "Respond with one short, sharp paragraph using religious, moral, or spiritual reasoning. "
                   "Be persuasive but concise."),
        ("user", "{messages}")
    ])
    chain = prompt | llm
    response = chain.invoke({"messages": state["messages"]})
    
    # log message
    state["messages"].append(f"Religion: {response}")
    
    # update next turn
    state["next"] = "evaluator"
    return state

import random

def evaluator_agent(state: DebateState):
    """Harsh evaluator that scores both sides and picks best highlights."""
    
    # grab last two debater messages
    last_science = [m for m in state["messages"] if m.startswith("Science:")][-1]
    last_religion = [m for m in state["messages"] if m.startswith("Religion:")][-1]
    
    # dummy scoring: could replace with LLM judgment, but here random for demo
    science_score = random.randint(5, 10)
    religion_score = random.randint(5, 10)
    
    state["scores"]["science"] += science_score
    state["scores"]["religion"] += religion_score
    
    # pick "highlight" (just choose whichever scored higher this round)
    if science_score >= religion_score:
        state["highlights"]["science"] = last_science
        highlight = f"Best this round: Science side → {last_science}"
    else:
        state["highlights"]["religion"] = last_religion
        highlight = f"Best this round: Religion side → {last_religion}"
    
    # log judge decision
    decision = (f"Evaluator: Science scored {science_score}, "
                f"Religion scored {religion_score}. {highlight}")
    state["messages"].append(decision)
    
    # advance round or end debate
    if state["round"] >= state["max_rounds"]:
        state["next"] = END
    else:
        state["round"] += 1
        state["next"] = "science"
    
    return state

# --- workflow setup ---
workflow = StateGraph()
workflow.add_node("science", science_agent)
workflow.add_node("religion", religion_agent)
workflow.add_node("evaluator", evaluator_agent)
workflow.set_entry_point("science")

# --- user input ---
topic = input("Enter the debate topic (e.g., Education vs Money): ")

# sides assignment
sides = {
    "science": topic.split(" vs ")[0] if "vs" in topic else "Education",
    "religion": topic.split(" vs ")[1] if "vs" in topic else "Money"
}

# --- initial state ---
state = DebateState(
    messages=[],
    next=workflow.entry_point,
    round=1,
    max_rounds=5,
    topic=topic,
    sides=sides,
    scores={"science": 0, "religion": 0},
    highlights={"science": "", "religion": ""}
)

# --- execution loop ---
while state["next"] != END:
    agent_func = workflow.get_node(state["next"])
    state = agent_func(state)
    print(state["messages"][-1])   # print only the latest message

# --- final results ---
print("\n--- FINAL RESULTS ---")
print(f"Science total: {state['scores']['science']} points")
print(f"Religion total: {state['scores']['religion']} points")

if state["scores"]["science"] > state["scores"]["religion"]:
    print("WINNER: Science side")
elif state["scores"]["science"] < state["scores"]["religion"]:
    print("WINNER: Religion side")
else:
    print("DRAW!")

print("\n--- BEST HIGHLIGHTS ---")
print(f"Science: {state['highlights']['science']}")
print(f"Religion: {state['highlights']['religion']}")
