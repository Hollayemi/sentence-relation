from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


app = FastAPI()

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure you're using a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# input schema
class ProductInfo(BaseModel):
    prodPrice: str
    specification: dict
    category: str
    subcategory: str

# Function to generate product description
def generate_product_description(product_specs, category, subcategory, prod_price):
 
    prompt = (
        f"Generate a detailed product description for a {category} in the {subcategory} category, "
        f"priced at {prod_price}. The product has the following specifications: {product_specs}."
    )

    # Tokenize and encode the input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text
    outputs = model.generate(
        inputs, max_length=150, do_sample=True, top_k=50, top_p=0.95, temperature=0.7
    )

    # Decode the generated text
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description


@app.post("/generate-description")
def get_product_description(product: ProductInfo):
    prodPrice = product.prodPrice
    specification = product.specification
    category = product.category
    subcategory = product.subcategory

    description = generate_product_description(
        specification, category, subcategory, prodPrice
    )
    
    return {"product_description": description}
