from langgraph.graph import START, END, StateGraph
from langchain_ollama import OllamaLLM
from typing import TypedDict


model = OllamaLLM(model='llama3.2')

class Input(TypedDict):
    title: str
    outline: str
    content: str


def create_outline(state: Input) -> Input:
    prompt = f"Create a detailed outline for an article titled '{state['title']}'."
    outline = model.invoke(prompt)  
    state['outline'] = outline
    return state

def write_content(state: Input) -> Input:
    prompt = f"Write a detailed article based on the following outline:\n{state['outline']}"
    content = model.invoke(prompt)
    state['content'] = content
    return state


graph = StateGraph(Input)

graph.add_node('create_outline', create_outline)
graph.add_node('write_content', write_content)

graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'write_content')
graph.add_edge('write_content', END)

workflow = graph.compile()


initial_state = {'title': 'The Future of AI in Everyday Life', 'outline': '', 'content': ''}

final_state = workflow.invoke(initial_state)

print(final_state['content'])