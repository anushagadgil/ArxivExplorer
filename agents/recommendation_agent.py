import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

class RecommendationAgent:
    def __init__(self, db_path="db/metadata.db", faiss_path="index/faiss_index.bin", model_name="all-MiniLM-L6-v2"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()
        self.index = faiss.read_index(faiss_path)
        
        # Force CPU
        self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def recommend(self, query, top_k=5):
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Embed the query on CPU
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # Search top candidates via FAISS
        faiss_top_k = 20
        distances, indices = self.index.search(query_embedding, faiss_top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            self.cur.execute("SELECT title, abstract, categories, authors, update_date FROM papers WHERE id=?", (int(idx),))
            row = self.cur.fetchone()

            if row:
                # Re-encode the paper text on CPU
                paper_embedding = self.model.encode(
                    [row["title"] + " " + row["abstract"]],
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )

                cos_score = util.cos_sim(torch.tensor(query_embedding[0]), paper_embedding[0]).item()

                results.append({
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "categories": row["categories"],
                    "authors": row["authors"],
                    "update_date": row["update_date"],
                    "cosine_similarity": round(cos_score, 4),
                    "faiss_distance": round(float(distance), 4)
                })

        results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return results[:top_k]