from transformers import pipeline
from rouge_score import rouge_scorer

class SummarizationAgent:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.summarizer = pipeline("summarization", model=model_name)
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def summarize(self, text, max_tokens=150):
        summary = self.summarizer(
            text,
            max_length=max_tokens,
            min_length=30,
            do_sample=False
        )[0]['summary_text']

        # Computing ROUGE-L against original text
        score = self.rouge.score(text, summary)['rougeL'].fmeasure
        return summary, round(score, 4)