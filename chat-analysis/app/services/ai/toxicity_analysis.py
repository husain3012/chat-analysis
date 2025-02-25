import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class ToxicityAnalysis:
    def __init__(self, model_name="unitary/unbiased-toxic-roberta", device=None):
        """
        Initializes the toxicity detection model.
        :param model_name: Name of the Hugging Face model for toxicity classification.
        :param device: 'cuda' for GPU, 'cpu' for CPU (auto-detect if not specified).
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)

        # Get label names
        self.labels = self.model.config.id2label

    def analyze(self, text):
        """
        Detects toxicity for a single text.
        :param text: Input text (can be None).
        :return: Dictionary with toxicity scores or None.
        """
        if text is None or not isinstance(text, str) or text.strip() == "":
            return None  # Handle missing or empty text

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        scores = outputs.logits.sigmoid().tolist()[0]  # Convert logits to probabilities
        toxicity_dict = {
            self.labels[i]: round(scores[i], 4) for i in range(len(scores))
        }
        toxicity_score = toxicity_dict["toxicity"]
        toxicity_dict.pop("toxicity")
        toxicity_type = None
        if toxicity_score > 0.5:
            toxicity_type = max(toxicity_dict, key=toxicity_dict.get)
        return toxicity_score, toxicity_type

    def analyze_batch(self, texts, batch_size=32):
        """
        Detects toxicity for a batch of texts.
        :param texts: List of input texts.
        :param batch_size: Number of texts per batch.
        :return: List of toxicity score dictionaries.
        """
        results = []
        total_batches = (
            len(texts) + batch_size - 1
        ) // batch_size  # Ensure correct batch size

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Processing Batches for toxicity analysis",
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

            batch_scores = (
                outputs.logits.sigmoid().tolist()
            )  # Convert logits to probabilities

            for text, scores in zip(processed_texts, batch_scores):
                if text is None:
                    results.append(None)
                    continue

                toxicity_dict = {
                    self.labels[i]: round(scores[i], 4) for i in range(len(scores))
                }
                toxicity_score = toxicity_dict["toxicity"]
                toxicity_dict.pop("toxicity")
                toxicity_type = None
                if toxicity_score > 0.5:
                    toxicity_type = max(toxicity_dict, key=toxicity_dict.get)
                detected_toxicity = toxicity_score, toxicity_type
                results.append(detected_toxicity)

        return results
