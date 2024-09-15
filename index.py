from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize FastAPI app
app = FastAPI()

# Create a request model to receive the sentences
class SentencesInput(BaseModel):
    sentence1: str
    sentence2: str

# Function to get similarity between two sentences
def get_similarity(sentence1, sentence2):
    inputs = tokenizer.encode_plus(sentence1, sentence2, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    similarity_score = torch.nn.functional.softmax(outputs.logits, dim=1)[0][1].item()
    return similarity_score

# Define the API endpoint
@app.post("/similarity")
async def compute_similarity(sentences: SentencesInput):
    similarity = get_similarity(sentences.sentence1, sentences.sentence2)
    return {"similarity_score": similarity}
