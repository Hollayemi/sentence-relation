from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load pre-trained model and tokenizer for sentence embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# FastAPI app initialization
app = FastAPI()

# Data model for request body
class Sentences(BaseModel):
    sentence1: str
    sentence2: str

# Define the similarity function using cosine similarity
def get_similarity(sentence1: str, sentence2: str) -> float:
    # Tokenize and encode the sentences
    inputs1 = tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True)
    inputs2 = tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True)
    
    # Get embeddings from the model
    with torch.no_grad():
        embedding1 = model(**inputs1).last_hidden_state.mean(dim=1)  # Mean pooling
        embedding2 = model(**inputs2).last_hidden_state.mean(dim=1)  # Mean pooling

    # Calculate cosine similarity between the two embeddings
    similarity_score = F.cosine_similarity(embedding1, embedding2).item()
    
    # Convert cosine similarity score (-1 to 1) to a percentage (0 to 100)
    percentage_similarity = (similarity_score + 1) / 2 * 100
    return percentage_similarity

# API route to calculate sentence similarity
@app.post("/similarity")
def calculate_similarity(sentences: Sentences):
    similarity = get_similarity(sentences.sentence1, sentences.sentence2)
    return {"similarity_percentage": similarity}
