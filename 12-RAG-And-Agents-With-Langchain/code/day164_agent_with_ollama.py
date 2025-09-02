import langchain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from day164_vector import retriever

model = OllamaLLM(model = "llama3.2")

template = """
You are an expert in answering question about pizza resturant

Here are some relevant reviews: {Review}

Here's the question to answer: {question}

"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    question = input("What do you want to ask about the resturant? (q to quit) ")
    if question == "q":
        break 

    reviews  = retriever.invoke(question)
    result = chain.invoke({"Review": [], "question": question})
    print(result)
