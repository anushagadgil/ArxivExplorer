import networkx as nx
from sentence_transformers import SentenceTransformer, util

class KnowledgeGraphMerged:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def build_graph(self, papers, cluster_labels=None):
        """
        Build a merged knowledge graph:
        - Category nodes from metadata 
        - Cluster assignment used if no category
        - Edges created based on category/cluster or cosine similarity
        """
        G = nx.Graph()
        texts = []

        for i, paper in enumerate(papers):
            title = paper['title']
            category = paper.get("categories", "").split()[0] if paper.get("categories") else None
            cluster = cluster_labels[i] if cluster_labels else None

            # Prefer category, fallback to cluster
            topic_label = category if category else f"Cluster {cluster}"

            # Adding a topic node if required
            if not G.has_node(topic_label):
                G.add_node(topic_label, type="topic")

            # Add paper node and edge to topic
            G.add_node(title, type="paper", authors=paper['authors'], abstract=paper['abstract'], topic=topic_label)
            G.add_edge(title, topic_label, relation="belongs_to")

            texts.append(paper['title'] + " " + paper['abstract'])

        # Step 2: Computing cosine similarities
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim > 0.75:
                    G.add_edge(papers[i]['title'], papers[j]['title'], weight=sim)

        return G
