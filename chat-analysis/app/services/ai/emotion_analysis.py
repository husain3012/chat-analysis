import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

emotions = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


class EmotionAnalysis:
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions", device=None):
        """
        Initializes the emotion detection model.
        :param model_name: Hugging Face model for emotion classification.
        :param device: 'cuda' for GPU, 'cpu' for CPU (auto-detect if not specified).
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)

        # Get emotion labels
        self.labels = self.model.config.id2label
        self.n_out_labels = 3

    def analyze(self, text):
        """
        Detects emotion for a single text.
        :param text: Input text (can be None).
        :return: Dictionary with emotion scores or None.
        """
        if text is None or not isinstance(text, str) or text.strip() == "":
            return None  # Handle missing or empty text

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        scores = outputs.logits.softmax(dim=1).tolist()[
            0
        ]  # Convert logits to probabilities
        emotion_dict = {self.labels[i]: round(scores[i], 4) for i in range(len(scores))}

        top_emotions = sorted(emotion_dict, key=emotion_dict.get, reverse=True)[
            : self.n_out_labels
        ]
        # emotions with score
        emotion_scores = {emotion: emotion_dict[emotion] for emotion in top_emotions}

        return emotion_scores

    def analyze_batch(self, texts, batch_size=32):
        """
        Detects emotion for a batch of texts.
        :param texts: List of input texts.
        :param batch_size: Number of texts per batch.
        :return: List of emotion score dictionaries.
        """
        results = []
        total_batches = (
            len(texts) + batch_size - 1
        ) // batch_size  # Ensure correct batch size

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Processing Batches for Emotion Analysis",
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

            batch_scores = outputs.logits.softmax(
                dim=1
            ).tolist()  # Convert logits to probabilities

            for text, scores in zip(processed_texts, batch_scores):
                if text is None:
                    results.append(None)
                    continue

                emotion_dict = {
                    self.labels[i]: round(scores[i], 4) for i in range(len(scores))
                }
                # top 3 emotions
                top_emotions = sorted(emotion_dict, key=emotion_dict.get, reverse=True)[
                    : self.n_out_labels
                ]
                # emotions with score
                emotion_scores = {
                    emotion: emotion_dict[emotion] for emotion in top_emotions
                }
                results.append(emotion_scores)

        return results
