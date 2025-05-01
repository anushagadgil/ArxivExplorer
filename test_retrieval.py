from agents.summarization_agent import SummarizationAgent

sample_text = """
Quantum Computing is a new and exciting field at the intersection of mathematics,
computer science and physics. It concerns a utilization of quantum mechanics
to improve the efficiency of computation. Here we present a gentle introduction
to some of the ideas in quantum computing. The paper begins by explaining...
"""

agent = SummarizationAgent()
summary, rouge_score = agent.summarize(sample_text)

print("📝 Summary:\n", summary)
print(f"\n📏 ROUGE-L Score (vs original): {rouge_score}")