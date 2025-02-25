import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class SentimentAnalysis:
    def __init__(
        self, model_name="tabularisai/multilingual-sentiment-analysis", device=None
    ):
        """
        Initializes the sentiment analysis model.
        :param model_name: Hugging Face model name.
        :param device: 'cuda' for GPU, 'cpu' for CPU (auto-detect if not specified).
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)

        # Define sentiment labels (Modify based on the chosen model)
        self.labels = self.model.config.id2label

    def analyze(self, text):
        """
        Analyzes sentiment for a single text.
        :param text: Input text (can be None).
        :return: Sentiment label (string) or None if input is invalid.
        """
        if text is None or not isinstance(text, str) or text.strip() == "":
            return None  # Return None if text is missing or empty

        input_text = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**input_text)

        sentiment_score = output.logits.softmax(dim=-1)
        sentiment_label = torch.argmax(sentiment_score, dim=-1).item()

        return self.labels.get(sentiment_label, "Unknown")

    def analyze_batch(self, texts, batch_size=32):
        """
        Analyzes sentiment for a batch of texts.
        :param texts: List of input texts.
        :param batch_size: Number of texts per batch.
        :return: List of sentiment labels.
        """
        results = []
        total_batches = (
            len(texts) + batch_size - 1
        ) // batch_size  # Ensure correct batch size

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Processing Batches for Sentiment Analysis",
        ):
            batch_texts = texts[i : i + batch_size]

            # Handle None or empty strings
            processed_texts = [
                text if isinstance(text, str) and text.strip() != "" else None
                for text in batch_texts
            ]

            if all(
                t is None for t in processed_texts
            ):  # If all texts are None, return a batch of None
                results.extend([None] * len(batch_texts))
                continue

            inputs = self.tokenizer(
                [t if t is not None else "" for t in processed_texts],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            for text, logits in zip(processed_texts, outputs.logits):
                if text is None:
                    results.append(None)  # If text is None, return None
                    continue

                sentiment_score = logits.softmax(dim=-1)
                sentiment_label = torch.argmax(sentiment_score, dim=-1).item()

                results.append(self.labels.get(sentiment_label, "Unknown"))

        return results
