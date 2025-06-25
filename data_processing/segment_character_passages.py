import re
import json
import os
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

# Step 1: Download the book if not already present
BOOK_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
BOOK_FILE = "pride_and_prejudice.txt"

if not os.path.exists(BOOK_FILE):
    import urllib.request
    urllib.request.urlretrieve(BOOK_URL, BOOK_FILE)

# Step 2: Load and trim book content
with open(BOOK_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

# Use regex to find start and end markers
start_marker_match = re.search(r'CHAPTER\s+I', full_text)
start_index = start_marker_match.start() if start_marker_match else 0
end_index = full_text.lower().find("end of the project gutenberg")

book_text = full_text[start_index:end_index].strip()

print("✅ Book trimmed successfully.")
print("Characters in trimmed book:", len(book_text))

# Step 3: Split into paragraphs (allowing for varied line breaks)
paragraphs = [p.strip() for p in re.split(r'\n\s*\n', book_text) if p.strip()]
print(f"Total paragraphs found: {len(paragraphs)}")

# Step 4: Define matchers for character mentions and pronouns
def mentions_elizabeth(paragraph):
    return bool(re.search(r'\b(elizabeth|miss bennet)\b', paragraph, re.IGNORECASE))

def mentions_darcy(paragraph):
    return bool(re.search(r'\b(darcy|mr\. darcy)\b', paragraph, re.IGNORECASE))

def references_pronouns(paragraph):
    return bool(re.search(r'\b(she|her|he|him|his|they|them|their)\b', paragraph, re.IGNORECASE))

# Step 5: Extract variable-length chunks
def extract_chunks(character_name, mention_fn):
    chunks = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        if mention_fn(para):
            chunk = [para]
            word_count = len(word_tokenize(para))
            j = i + 1

            # Expand forward to collect more context
            while j < len(paragraphs) and word_count < 250:
                next_para = paragraphs[j]
                if mention_fn(next_para) or references_pronouns(next_para):
                    chunk.append(next_para)
                    word_count += len(word_tokenize(next_para))
                    j += 1
                else:
                    break

            # Save chunk
            full_chunk = " ".join(chunk)
            chunks.append({
                "character": character_name,
                "text": full_chunk
            })

            i = j  # skip ahead to avoid overlap
        else:
            i += 1
    return chunks

# Step 6: Extract for both characters
elizabeth_chunks = extract_chunks("Elizabeth Bennet", mentions_elizabeth)
darcy_chunks = extract_chunks("Mr. Darcy", mentions_darcy)

# Step 7: Save outputs
os.makedirs("output", exist_ok=True)
with open("output/elizabeth_passages.json", "w") as f:
    json.dump(elizabeth_chunks, f, indent=2)
with open("output/darcy_passages.json", "w") as f:
    json.dump(darcy_chunks, f, indent=2)

print(f"\n✅ Done.")
print(f"Extracted {len(elizabeth_chunks)} Elizabeth chunks and {len(darcy_chunks)} Darcy chunks.")

