import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import re

class LanguageTranslation:
    def __init__(
        self,

        model_name="rudrashah/RLM-hinglish-translator",
        device=None,
    ):
        """
        Initializes the language detection model.
        :param languages: List of valid languages to consider.
        :param model_name: Hugging Face model name.
        :param device: 'cuda' for GPU, 'cpu' for CPU (auto-detect if not specified).
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            self.device
        )
        # WITHOUT THIS IT WILL GIVE RANDOM OUTPUT
        self.template = "Hinglish:\n{hi_en}\n\nEnglish:\n{en}"

        self.output_regex = r"<bos>Hinglish:\n(.*)\nEnglish:\n(.*)"

    def translate(self, text):
        """

        :param text: Input text (can be None).
        :return: Detected language (string) or None if text is None.
        """
        if text is None or not isinstance(text, str) or text.strip() == "":
            return None  # Return None if text is missing or empty

        input_text = self.tokenizer(self.template.format(hi_en=text,en=""),return_tensors="pt")
        output = self.model.generate(**input_text)
        translated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # get group 2 from the regex match
        translated_text = re.match(self.output_regex, translated_text).group(2)
        return translated_text
        

    def translate_batch(self, texts, batch_size=32):
        """
     
        :param texts: List of input texts.
        :param batch_size: Number of texts to process per batch.
        :return: List of detected languages.
        """
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
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
                processed_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                decoded_outputs = self.tokenizer.batch_decode(
                    outputs.logits, skip_special_tokens=True
                )
                outputs = [re.match(self.output_regex, text).group(2) for text in decoded_outputs]
                results.extend(outputs)
                

        return results

    def translate_language_df(
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
        # for text in chat_df where language is not english, translate to english

        chat_df["translated"] = chat_df[text_column]
        chat_df.loc[chat_df[language_column] != "en", "translated_text"] = self.translate_batch(
            chat_df.loc[chat_df[language_column] != "en", text_column].tolist(),
            batch_size=batch_size,
        )
        return chat_df
        
        
