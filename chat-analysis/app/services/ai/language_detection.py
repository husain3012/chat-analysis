import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd


class LanguageDetection:
    def __init__(
        self,
        languages=["hi", "en"],
        model_name="papluca/xlm-roberta-base-language-detection",
        device=None,
    ):
        """
        Initializes the language detection model.
        :param languages: List of valid languages to consider.
        :param model_name: Hugging Face model name.
        :param device: 'cuda' for GPU, 'cpu' for CPU (auto-detect if not specified).
        """
        self.valid_languages = languages
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)

    def detect_language(self, text):
        """
        Detect the language of a single text.
        :param text: Input text (can be None).
        :return: Detected language (string) or None if text is None.
        """
        if text is None or not isinstance(text, str) or text.strip() == "":
            return None  # Return None if text is missing or empty

        input_text = self.tokenizer(text, return_tensors="pt").to(self.device)
        output = self.model(**input_text)

        # Get language labels
        languages = [
            self.model.config.id2label[i]
            for i in range(len(self.model.config.id2label))
        ]
        language_with_scores_dict = dict(zip(languages, output.logits[0].tolist()))

        # Filter valid languages
        valid_language_with_scores_dict = {
            k: v
            for k, v in language_with_scores_dict.items()
            if k in self.valid_languages
        }

        # Return highest probability language or None if empty
        return max(
            valid_language_with_scores_dict,
            key=valid_language_with_scores_dict.get,
            default=None,
        )

    def detect_language_batch(self, texts, batch_size=32):
        """
        Detect language for a batch of texts.
        :param texts: List of input texts.
        :param batch_size: Number of texts to process per batch.
        :return: List of detected languages.
        """
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Detecting languages"):
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

                languages = [
                    self.model.config.id2label[i]
                    for i in range(len(self.model.config.id2label))
                ]
                language_with_scores_dict = dict(zip(languages, logits.tolist()))

                # Filter only valid languages
                valid_language_with_scores_dict = {
                    k: v
                    for k, v in language_with_scores_dict.items()
                    if k in self.valid_languages
                }

                # Get the highest probability language or None
                detected_language = max(
                    valid_language_with_scores_dict,
                    key=valid_language_with_scores_dict.get,
                    default=None,
                )
                results.append(detected_language)

        return results

    def detect_language_df(
        self,
        chat_df: pd.DataFrame,
        text_column="content",
        language_column="language",
        batch_size=32,
    ):
        """
        Detect language for a DataFrame with text content.
        :param chat_df: Input DataFrame with text content.
        :param text_column: Column name containing text content.
        :return: DataFrame with detected language column.
        """
        chat_df[language_column] = self.detect_language_batch(
            chat_df[text_column], batch_size=batch_size
        )
        return chat_df
