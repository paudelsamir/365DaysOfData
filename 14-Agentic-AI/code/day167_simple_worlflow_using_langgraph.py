from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama


model = ChatOllama(model="llama3.2")
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": add_messages(messages, [response])}

graph = StateGraph(dict)
graph.add_node("chatbot", call_model)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

state = { "messages": [SystemMessage(content="You are a good friend of mine, you are going to talk with me like a best friend in short")] }

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break
    state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])
    state = app.invoke(state)
    print("AI:", state["messages"][-1].content)


print(app.get_graph().draw_mermaid())