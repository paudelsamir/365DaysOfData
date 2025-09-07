from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import datetime

model = ChatOllama(model="llama3.2")


def calculator_tool(state):
    user_message = state["user_message"].content
    try:
        result = eval(user_message)
        reply = f"The result of {user_message} is {result}."
    except Exception as e:
        reply = f"Error in calculation: {str(e)}"
    
    return {"messages": add_messages(state["messages"], [AIMessage(content=reply)])}
    

def time_tool(state):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"messages": add_messages(state["messages"], [AIMessage(content=f"The current date and time is: {current_time}")])}


def smalltalk_node(state):
    response = model.invoke(state["messages"])

    return {"messages": add_messages(state["messages"], [AIMessage(content=response.content)])}

def router(state):
    last_user_message = state["user_message"].content.lower()
    if any(keyword in last_user_message for keyword in ["calculate", "what is", "solve", "+", "-", "*", "/"]):
        return "calculator"
    elif any(keyword in last_user_message for keyword in ["time", "date", "day", "now"]):
        return "time"
    else:
        return "smalltalk"

graph = StateGraph(dict)
graph.add_node("calculator", calculator_tool)
graph.add_node("time", time_tool)
graph.add_node("smalltalk", smalltalk_node)
graph.add_conditional_edges(

    START, router, 
    {
        "calculator": "calculator",
        "time": "time",
        "smalltalk": "smalltalk"
    }
)
graph.add_edge("calculator", END)
graph.add_edge("time", END)
graph.add_edge("smalltalk", END)
app = graph.compile()
state = {"messages": []}
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting chat.")
        break
    state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])
    state["user_message"] = HumanMessage(content=user_input) 
    state = app.invoke(state)
    print("AI:", state["messages"][-1].content)