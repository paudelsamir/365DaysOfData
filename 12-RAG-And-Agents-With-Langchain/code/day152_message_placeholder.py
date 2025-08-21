from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

chat_template = ChatPromptTemplate([
    ('system', 'You are my helpful personal assistant.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

history_file = 'chat_history.txt'
if os.path.exists(history_file):
    with open(history_file) as f:
        chat_history = [line.strip() for line in f if line.strip()]

print("Current chat history:", chat_history)

query = 'Remind me to call mom tomorrow.'

prompt = chat_template.invoke({'chat_history': chat_history, 'query': query})

print(prompt)
