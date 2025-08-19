from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Kathmandu is the capital of Nepal",
    "Mount Everest is located in Nepal",
    "Nepal is famous for its beautiful mountains"
]

vector = embedding.embed_documents(documents)

print(str(vector))