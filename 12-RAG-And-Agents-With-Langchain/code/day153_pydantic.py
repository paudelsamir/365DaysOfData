# from pydantic import BaseModel, EmailStr, Field
# from typing import Optional

# class Student(BaseModel):
#     name: str = 'sam'
#     age: Optional[int] = None
#     email: EmailStr
#     cgpa: float = Field(gt=0, lt=10, default=7.5, description='A decimal value representing the cgpa of the student')

# new_student = {'age': '25', 'email': 'sam@example.com'}

# student = Student(**new_student)

# student_dict = student.model_dump()

# print(student_dict['cgpa'])

# student_json = student.model_dump_json()
# print(student_json)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

# schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="List all key themes discussed in the review")
    summary: str = Field(description="Brief summary of the review")
    sentiment: Literal["pos", "neg", "neutral"] = Field(description="Sentiment of the review: pos, neg, or neutral")
    pros: Optional[list[str]] = Field(default=None, description="List of pros")
    cons: Optional[list[str]] = Field(default=None, description="List of cons")
    name: Optional[str] = Field(default=None, description="Name of the reviewer")

structured_model = model.with_structured_output(Review)

review_text = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by samireey
"""

result = structured_model.invoke(review_text)

print(result)