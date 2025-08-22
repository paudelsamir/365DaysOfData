# {
#     "title": "student",
#     "description": "schema about students",
#     "type": "object",
#     "properties":{
#         "name":"string",
#         "age":"integer"
#     },
#     "required":["name"]
# }


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

# schema
json_schema = {
    "title": "Student",
    "description": "Schema about students",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The name of the student"
        },
        "age": {
            "type": "integer",
            "description": "The age of the student"
        }
    },
    "required": ["name"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""
Name: Alice Johnson
Age: 21
Alice is a computer science student who enjoys coding, participating in hackathons, and learning about artificial intelligence.
""")

print(result)
