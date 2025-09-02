from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd


df = pd.read_csv("/home/sam/Github/365DaysOfData/12-RAG-And-Agents-With-Langchain/code/realistic_restaurant_reviews.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location  = "/home/sam/chroma_db"

add_documents = not os.path.exists(db_location) or len(os.listdir(db_location)) == 0 


if add_documents:
    documents = []
    ids = []


    for i, row in df.iterrows():
        docuemnts = Document(page_content=row["Review"], metadata={"index": i}, id= str(i))
        documents.append(docuemnts)
        ids.append(str(i))
    

vector_store = Chroma(collection_name='resturant_reviews', persist_directory=db_location, embedding_function=embeddings)


if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})