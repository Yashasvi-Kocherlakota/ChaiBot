from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

passages_dir = "data_processing/data"

class ChatRequest(BaseModel):
    character: str
    message: str

def load_passages(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load data
elizabeth_passages = load_passages(os.path.join(passages_dir, "elizabeth_passages.json"))
darcy_passages = load_passages(os.path.join(passages_dir, "darcy_passages.json"))
index = faiss.read_index(os.path.join(passages_dir, "index.faiss"))
all_passages = elizabeth_passages + darcy_passages

@app.post("/chat")
def chat(request: ChatRequest):
    character = request.character
    message = request.message

    # Step 1: Embed the user's message
    embedding = model.encode([message])

    # Step 2: Search for similar passages
    D, I = index.search(embedding, k=5)

    for i in I[0]:
        passage = all_passages[i]
        if character.lower() in passage["character"].lower():
            raw_text = passage["text"]

            # Step 3: Use GPT to generate a natural reply
            prompt = (
                f"You are {character} from *Pride and Prejudice*. "
                f"Respond to the following question in your own voice.\n\n"
                f"Question: {message}\n\n"
                f"Here is some context from the original book that may help:\n\"{raw_text}\"\n\n"
                f"Now reply as {character} would:"
            )

            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            reply = completion.choices[0].message.content
            return {"response": reply}

    return {"response": "No relevant passage found for this character."}
