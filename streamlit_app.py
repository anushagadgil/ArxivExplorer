import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from agents.recommendation_agent import RecommendationAgent
from agents.summarization_agent import SummarizationAgent
from agents.semantic_clustering_agent import SemanticClusteringAgent
from agents.knowledge_graph_from_clusters import KnowledgeGraphMerged

# Initialize agents
recommendation_agent = RecommendationAgent()
summarization_agent = SummarizationAgent()
cluster_agent = SemanticClusteringAgent(n_clusters=5)
graph_agent = KnowledgeGraphMerged()

# UI setup
st.title("📚 Academic Research Assistant")
st.markdown("Suggests research papers, summarizes, and builds contextual + topic graphs.")

query = st.text_input("🔍 Enter a topic or query to get recommended papers:")

if query:
    try:
        # Get paper recommendations
        results = recommendation_agent.recommend(query, top_k=5)

        if results:
            st.subheader("📄 Top Recommended Papers:")
            for i, paper in enumerate(results):
                st.markdown(f"**{i+1}. {paper['title']}**")
                st.markdown(f"_Authors_: {paper['authors']}")
                st.markdown(f"_Category_: `{paper['categories']}`")
                st.markdown(f"_Cosine Similarity_: {paper['cosine_similarity']}  |  _FAISS Distance_: {paper['faiss_distance']}")
                st.markdown(f"_Abstract Preview_: {paper['abstract'][:400]}...")

                # Summarization
                with st.expander("📝 Click to summarize"):
                    summary, rouge = summarization_agent.summarize(paper['abstract'])
                    st.markdown(f"**Summary:** {summary}")
                    st.markdown(f"📏 ROUGE-L Score: {rouge}")
                st.markdown("---")

            # Graph building
            st.subheader("🧠 Knowledge Graph (Contextual + Categorical)")
            clustered_papers = cluster_agent.cluster_papers(results)
            G = graph_agent.build_graph(clustered_papers)

            # Visualize the graph
            fig, ax = plt.subplots(figsize=(10, 6))
            pos = nx.spring_layout(G, seed=42)
            topic_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "topic"]
            paper_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "paper"]

            nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_color='skyblue', node_size=900, label='Topics')
            nx.draw_networkx_nodes(G, pos, nodelist=paper_nodes, node_color='lightgreen', node_size=600, label='Papers')
            nx.draw_networkx_edges(G, pos, alpha=0.4)
            nx.draw_networkx_labels(G, pos, font_size=8)

            st.pyplot(fig)

        else:
            st.warning("No relevant papers found. Try refining your query.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
