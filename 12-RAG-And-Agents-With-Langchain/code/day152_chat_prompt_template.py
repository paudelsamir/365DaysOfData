from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are an expert in {domain}.'),
    ('human', 'Can you help me understand {topic}?'),
])

prompt = chat_template.invoke({'domain':'mythopoeia', 'topic':'serendipity'})

print(prompt)