from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_similarity(sentence1, sentence2):
    inputs = tokenizer.encode_plus(sentence1, sentence2, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    similarity_score = torch.nn.functional.softmax(outputs.logits, dim=1)[0][1].item()
    return similarity_score

sentence1 = "The quick brown fox jumps over the lazy dog."
sentence2 = "A fast brown fox leaps over a sleepy dog."
similarity = get_similarity(sentence1, sentence2)
print(f"Similarity: {similarity}")

