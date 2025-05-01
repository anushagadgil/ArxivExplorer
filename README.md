# ArxivExplorer

ArxivExplorer is an agentic AI system that allows users to search, explore, and understand research papers from the arXiv dataset. It combines semantic search, intelligent summarization, and topic-based knowledge graph generation to enhance research discovery.

##  Features

-  **Semantic Paper Recommendation**: Finds top relevant papers using FAISS and Sentence Transformers based on a user query.
-  **Summarization Agent**: Summarizes abstracts with ROUGE-L scoring to indicate quality.
-  **Knowledge Graph Visualization**: Builds contextual and category-based graphs of research papers using NetworkX.
-  **Streamlit Interface**: Clean UI for searching, visualizing, and interacting with research papers.

## Tech Stack

- Python
- FAISS + SQLite (retrieval)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- NetworkX
- Transformers (for summarization)
- Streamlit (interface)

## File Structure

ArxivExplorer/
│
├── agents/
│   ├── recommendation_agent.py
│   ├── summarization_agent.py
│   ├── semantic_clustering_agent.py
│   └── knowledge_graph_merged.py
│
├── data/
│   └── arxiv-metadata-oai-snapshot.json
├── db/
│   └── metadata.db
├── index/
│   └── faiss_index.bin
├── build_faiss_and_sqlite.py
├── main.py
├── streamlit_app.py
├── .env
└── requirements.txt

## Getting Started

```bash
pip install -r requirements.txt
python build_faiss_and_sqlite.py    # Used for building vectorDB
streamlit run streamlit_app.py      # To launch UI
'''

# Notes
'''
Make sure your .env includes the correct path to the dataset:

DATA_PATH=data/arxiv-metadata-oai-snapshot.json

'''