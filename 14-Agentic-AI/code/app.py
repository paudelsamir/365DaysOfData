import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
import os
import datetime

# Initialize session state to store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = ""
if "command_executed" not in st.session_state:
    st.session_state.command_executed = False
if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"chat_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# App title and description
st.title("Bhagavad Gita Expert")
st.subheader("Ask questions about the Bhagavad Gita")
st.write("Type '/forget' to clear conversation history, or '/exit' to quit")

# Initialize components on first run
@st.cache_resource
def initialize_components():
    # Initialize LLM
    llm = OllamaLLM(model='llama3.2')
    
    # Load document
    pdf_path = "/home/sam/Github/365DaysOfData/14-Agentic-AI/code/extras/geeta.pdf"
    doc = PyPDFLoader(pdf_path)
    docs = doc.load()
    
    # Initialize embeddings and vector store
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    
    # Create parser
    parser = StrOutputParser()
    
    # Create prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Bhagavad Gita expert. Answer the question based on the context provided. If the context does not help, say 'I don't know'"),
        ("human", "query: {query} context: {context}"),
    ])
    
    # Define state functions
    def retrieve_context(state):
        query = state["messages"][-1].content if state["messages"] else ""
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        return {"context": context}

    def response_generation(state):
        context = state.get("context", "")
        query = state["messages"][-1].content if state["messages"] else ""

        # Get LLM response
        response = llm.invoke(chat_prompt.format_messages(query=query, context=context))
        
        # Handle response type
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
            
        return {"messages": state["messages"] + [AIMessage(content=response_text)]}

    def handle_commands(state):
        last_message = state["messages"][-1].content.lower()
        
        if last_message.startswith("/forget"):
            return {"messages": [state["messages"][-1]], "command_executed": True}
        return {"command_executed": False}
    
    # Define state type and create graph
    class State(TypedDict):
        messages: list
        context: str
        command_executed: bool

    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("command_handler", handle_commands)
    graph.add_node("respond", response_generation)
    
    # Connect nodes
    graph.add_edge(START, "command_handler")
    graph.add_conditional_edges(
        "command_handler",
        lambda state: state["command_executed"],
        {False: "retrieve", True: END}
    )
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)
    
    # Compile workflow
    workflow = graph.compile(checkpointer=MemorySaver())
    
    return {
        "llm": llm,
        "vectorstore": vectorstore,
        "parser": parser,
        "chat_prompt": chat_prompt,
        "workflow": workflow
    }

# Initialize components
components = initialize_components()
st.session_state.workflow = components["workflow"]

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
user_input = st.chat_input("Ask a question about the Bhagavad Gita...")

# Process user input
if user_input:
    # Add user message to chat
    with st.chat_message("user"):
        st.write(user_input)
    
    # Special command handling
    if user_input.lower() == "/forget":
        st.session_state.messages = []
        with st.chat_message("assistant"):
            st.write("Memory cleared! Let's start a new conversation.")
    else:
        # Create state for LangGraph
        state = {
            "messages": add_messages(st.session_state.messages, [HumanMessage(content=user_input)]), 
            "context": st.session_state.context, 
            "command_executed": st.session_state.command_executed
        }
        
        # Process with workflow
        with st.spinner("Thinking..."):
            state = st.session_state.workflow.invoke(
                state,
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id,
                        "checkpoint_ns": "bhagavad_gita_chat"
                    }
                }
            )
        
        # Update session state
        st.session_state.messages = state["messages"]
        st.session_state.context = state.get("context", "")
        st.session_state.command_executed = state.get("command_executed", False)
        
        # Display AI response
        with st.chat_message("assistant"):
            st.write(state["messages"][-1].content)