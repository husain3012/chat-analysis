import pandas as pd
from transformers import pipeline

class DeepSeek:
    def __init__(self, model_path="deepseek-model-path"):
        """
        Initialize the DeepSeek analyzer with a locally running LLM.
        """
        self.analyzer = pipeline("text-generation", model=model_path)

    def analyze_tone(self, text):
        """
        Analyze the overall tone of the conversation.
        """
        prompt = f"Analyze the tone of this text: {text}"
        response = self.analyzer(prompt, max_length=50)
        return response[0]["generated_text"].strip()

    def analyze_topics(self, text):
        """
        Extract key topics from the conversation.
        """
        prompt = f"Extract key topics from this text: {text}"
        response = self.analyzer(prompt, max_length=100)
        return response[0]["generated_text"].strip().split(", ")

    def analyze_relationships(self, participants, text):
        """
        Analyze relationships between participants.
        """
        prompt = f"Analyze the relationships between {', '.join(participants)} based on this text: {text}"
        response = self.analyzer(prompt, max_length=150)
        return response[0]["generated_text"].strip()

    def analyze_language(self, text):
        """
        Detect the language(s) used in the conversation.
        """
        prompt = f"Detect the language(s) used in this text and return their ISO 639-1 codes: {text}"
        response = self.analyzer(prompt, max_length=50)
        return response[0]["generated_text"].strip()

    def analyze_humor_level(self, text):
        """
        Analyze the humor level in the conversation.
        """
        prompt = f"Analyze the humor level in this text (low, medium, high): {text}"
        response = self.analyzer(prompt, max_length=50)
        return response[0]["generated_text"].strip()

    def analyze_cultural_backgrounds(self, participants, text):
        """
        Infer cultural backgrounds of participants.
        """
        prompt = f"Infer the cultural backgrounds of {', '.join(participants)} based on this text: {text}"
        response = self.analyzer(prompt, max_length=150)
        return response[0]["generated_text"].strip()

    def analyze_llm_insights(self, text):
        """
        Generate additional insights that can only be calculated via LLM.
        """
        prompt = f"Generate unique insights about this conversation: {text}"
        response = self.analyzer(prompt, max_length=200)
        return response[0]["generated_text"].strip()

    def analyze(self, df, participants):
        """
        Analyze the preprocessed dataframe of messages and return insights.
        """
        # Combine all messages into a single text for analysis
        combined_text = " ".join(df["content"].tolist())

        # Perform analysis
        insights = {
            "tone": self.analyze_tone(combined_text),
            "topics": self.analyze_topics(combined_text),
            "relationships": self.analyze_relationships(participants, combined_text),
            "language_used": self.analyze_language(combined_text),
            "humor_level": self.analyze_humor_level(combined_text),
            "cultural_backgrounds": self.analyze_cultural_backgrounds(participants, combined_text),
            "llm_insights": self.analyze_llm_insights(combined_text)
        }

        return insights