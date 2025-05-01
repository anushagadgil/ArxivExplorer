from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class SemanticClusteringAgent:
    def __init__(self, n_clusters=10, model_name="all-MiniLM-L6-v2"):
        self.n_clusters = n_clusters
        self.model = SentenceTransformer(model_name)

    def cluster_papers(self, papers):
        texts = [paper['title'] + " " + paper['abstract'] for paper in papers]
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # ✅ Safeguard: Don't ask for more clusters than samples
        num_clusters = min(self.n_clusters, len(papers))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        for i, paper in enumerate(papers):
            paper['cluster'] = labels[i]

        return papers