from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2")

# node 1: reasoning
def reasoning_node(state):
    messages = state["messages"]
    response = model.invoke(
        [HumanMessage(content="think step by step, but don’t answer yet")] + messages
    )
    return {"messages": add_messages(messages, [AIMessage(content="THOUGHTS: " + response.content)])}

# node 2: answering
def answering_node(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": add_messages(messages, [response])}

graph = StateGraph(dict)
graph.add_node("reasoning", reasoning_node)
graph.add_node("answering", answering_node)
graph.add_edge(START, "reasoning")
graph.add_edge("reasoning", "answering")
graph.add_edge("answering", END)

app = graph.compile()
state = {"messages": [HumanMessage(content="why is the sky blue?")]}
result = app.invoke(state)
print(result["messages"][-1].content)

# visualize
print(app.get_graph().draw_mermaid())


# LANGGRAPH IMPLEMENTATION EXAMPLE
# from typing_extensions import TypedDict
# from typing import List
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
# from langchain_ollama import ChatOllama


# model = ChatOllama(model="llama3.2")
# def call_model(state):
#     messages = state["messages"]
#     response = model.invoke(messages)
#     return {"messages": add_messages(messages, [response])}

# graph = StateGraph(dict)
# graph.add_node("chatbot", call_model)
# graph.add_edge(START, "chatbot")
# graph.add_edge("chatbot", END)

# app = graph.compile()

# state = { "messages": [SystemMessage(content="You are a good friend of mine, you are going to talk with me like a best friend in short")] }

# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["exit", "quit"]:
#         print("Exiting the chat.")
#         break
#     state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])
#     state = app.invoke(state)
#     print("AI:", state["messages"][-1].content)


# print(app.get_graph().draw_mermaid())