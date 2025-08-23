# from typing import TypedDict

# class Car(TypedDict):
#     make: str
#     model: str
#     year: int

# new_car: Car = {'make': 'Toyota', 'model': 'Corolla', 'year': 2022}

# print(new_car)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "List all key themes discussed in the review"]
    summary: Annotated[str, "Provide a brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg", "neutral"], "Return sentiment of the review: positive, negative, or neutral"]
    pros: Annotated[Optional[list[str]], "List all pros"]
    cons: Annotated[Optional[list[str]], "List all cons"]
    name: Annotated[Optional[str], "Name of the reviewer"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by samireey
""")

print("Reviewer:", result.get('name'))
print("Summary:", result.get('summary'))
print("Sentiment:", result.get('sentiment'))
print("Key Themes:", result.get('key_themes'))
print("Pros:", result.get('pros'))
print("Cons:", result.get('cons'))