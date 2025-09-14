import cohere
import json
from fastapi import FastAPI
from pydantic import BaseModel, conlist

# Setup the Cohere client
co = cohere.ClientV2("j3dnErL21HLrrwoaSskJsQpmAn2SxRc1JzJWg7yL") # Get your free API key: https://dashboard.cohere.com/api-keys

# Step 2: Create a Sentiment Classifier
def classify_sentiment(product_review):
    # Create prompt with examples
    prompt = """Classify this text into positive, negative, or neutral sentiment. Here are some examples:

    Positive examples:
    - "The order came 5 days early"
    - "The item exceeded my expectations" 
    - "I ordered more for my friends"
    - "I would buy this again"
    - "I would recommend this to others"

    Negative examples:
    - "The package was damaged"
    - "The order is 5 days late"
    - "The order was incorrect" 
    - "I want to return my item"
    - "The item's material feels low quality"

    Neutral examples:
    - "The item was nothing special"
    - "I would not buy this again but it wasn't a waste of money"
    - "The item was neither amazing or terrible"
    - "The item was okay"
    - "I have no emotions towards this item"

    Text to classify:
    {}"""

    res = co.chat(
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": prompt.format(product_review)
            }
        ],
        temperature=0.0,
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "class": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"]
                    }
                },
                "required": ["class"]
            }
        }
    )
    return json.loads(res.message.content[0].text)["class"]

# Step 3: Create a FastAPI Endpoint
app = FastAPI()

class ProductReviews(BaseModel):
    reviews: conlist(str, min_length=1)

@app.post("/prediction")
def predict_sentiment(product_reviews: ProductReviews):
    sentiments = []
    for review in product_reviews.reviews:
        sentiments.append(classify_sentiment(review))
    return sentiments

# Step 4: Call the Endpoint
