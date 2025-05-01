import os
import json
import sqlite3
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths and constants
DATA_PATH = os.getenv("DATA_PATH", "data/arxiv-metadata-oai-snapshot.json")
MODEL_NAME = "all-MiniLM-L12-v2"  
DB_PATH = "db/metadata.db"
FAISS_INDEX_PATH = "index/faiss_index.bin"
MAX_PAPERS = 50000
BATCH_SIZE = 1000

# Create necessary folders
os.makedirs("db", exist_ok=True)
os.makedirs("index", exist_ok=True)

# Initialize model
print(f" Loading embedding model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()
print(f" Embedding dimension: {embedding_dim}")

# Set up SQLite
print(" Setting up SQLite database...")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute('''
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    categories TEXT,
    authors TEXT,
    update_date TEXT
)
''')
conn.commit()

# Init FAISS
index = faiss.IndexFlatL2(embedding_dim)

# Load raw lines
print("\n Reading dataset lines...")
with open(DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    lines = [line for _, line in zip(range(MAX_PAPERS * 2), f)]  

# Process papers
print(f"\n📚 Processing up to {MAX_PAPERS} valid papers...\n")
paper_id = 0
batch = []

for line in tqdm(lines, desc="Processing papers"):
    if paper_id >= MAX_PAPERS:
        break

    try:
        paper = json.loads(line)
        title = paper.get("title", "").strip()
        abstract = paper.get("abstract", "").strip()
        if not title or not abstract:
            continue

        combined = f"{title} {abstract}"
        batch.append((combined, title, abstract, paper.get("categories"), paper.get("authors"), paper.get("update_date")))

        if len(batch) >= BATCH_SIZE:
            texts = [b[0] for b in batch]
            embeddings = model.encode(texts, convert_to_numpy=True)
            index.add(embeddings)

            for _, title, abstract, categories, authors, update_date in batch:
                cur.execute('''
                    INSERT INTO papers (id, title, abstract, categories, authors, update_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (paper_id, title, abstract, categories, authors, update_date))
                paper_id += 1

            conn.commit()
            batch = []

    except json.JSONDecodeError:
        continue

# Final batch
if batch and paper_id < MAX_PAPERS:
    texts = [b[0] for b in batch]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index.add(embeddings)

    for _, title, abstract, categories, authors, update_date in batch:
        if paper_id >= MAX_PAPERS:
            break
        cur.execute('''
            INSERT INTO papers (id, title, abstract, categories, authors, update_date)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (paper_id, title, abstract, categories, authors, update_date))
        paper_id += 1

    conn.commit()

# Save index
faiss.write_index(index, FAISS_INDEX_PATH)
conn.close()

print(f"\n Done! {paper_id} papers processed.")
print(f" FAISS saved: {FAISS_INDEX_PATH}")
print(f" SQLite DB saved: {DB_PATH}")