import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Load the model (this downloads the MiniLM model)
print("🔍 Loading sentence-transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load passages
def load_passages(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("📚 Loading passage data...")
elizabeth_passages = load_passages("data_processing/data/elizabeth_passages.json")
darcy_passages = load_passages("data_processing/data/darcy_passages.json")
all_passages = elizabeth_passages + darcy_passages

texts = [p["text"] for p in all_passages]
labels = [p["character"] for p in all_passages]

# Generate embeddings
print("✨ Embedding passages...")
embeddings = model.encode(texts, show_progress_bar=True)
embedding_matrix = np.array(embeddings).astype("float32")

# Build FAISS index
print("📦 Building FAISS index...")
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Save index and metadata
faiss.write_index(index, "character_index.faiss")

with open("character_texts.json", "w") as f:
    json.dump(texts, f)

with open("character_labels.json", "w") as f:
    json.dump(labels, f)

print("✅ Done. Embeddings and index saved.")
