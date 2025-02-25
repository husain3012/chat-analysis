import pandas as pd
import emoji
from app.config import DATA_PATH
import json
from symspellpy import SymSpell, Verbosity
import pkg_resources
import nltk
import re
from tqdm import tqdm

tqdm.pandas()

# nltk.download("stopwords")
# nltk.download("punkt")

# Load the stopwords
from nltk.corpus import stopwords

# if stopwords are not downloaded, download them

STOP_WORDS = None
try:
    STOP_WORDS = set(stopwords.words())
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words())


class ChatPreprocessor:
    def __init__(self):
        """
        Preprocess chat data.
        """
        abbreviations_json = DATA_PATH / "static/abbreviations.json"
        self.abbreviations = {}

        with open(abbreviations_json, "r") as f:
            self.abbreviations = json.load(f)
            self.abbreviations = {
                k.lower(): v.lower() for k, v in self.abbreviations.items()
            }

    def demojize(self, chat_df: pd.DataFrame, columns=["content"]):
        """
        Convert emojis their text description
        :param columns: list of columns to process
        """
        for col in columns:
            chat_df[col] = chat_df[col].progress_apply(
                lambda x: (emoji.demojize(x) if type(x) == str else x)
            )
        return chat_df

    def expand_abbreviations(self, chat_df: pd.DataFrame, columns=["content"]):
        """
        Expands abbreviations in the chat data
        :param abbreviations: dictionary with abbreviations as keys and their expansions as values
        """
        for column in columns:
            chat_df[column] = chat_df[column].progress_apply(
                lambda x: (
                    " ".join(
                        [
                            self.abbreviations.get(
                                re.sub(r"[^a-zA-Z0-9\s]", "", word), word
                            )
                            for word in x.split()
                        ]
                    )
                    if type(x) == str
                    else x
                )
            )

        return chat_df

    def process(self, chat_df: pd.DataFrame, text_columns=["content"]):
        """
        Process chat data, end to end.
        Steps:
        1. Expand abbreviations
        2. Convert emojis to text

        """

        chat_df = self.expand_abbreviations(chat_df, columns=text_columns)
        chat_df = self.demojize(chat_df, columns=text_columns)
        return chat_df


class CleanTextForNLP:
    def __init__(self):
        """
        Cleans and corrects text using spell correction and remove special characters.
        :param dictionary_path: Path to a frequency-based dictionary file.
        :param max_edit_distance: Maximum allowed edit distance for suggestions.
        """
        self.sym_spell = SymSpell(max_dictionary_edit_distance=1)

        # dictionary_path = pkg_resources.resource_filename(
        #     "symspellpy", "frequency_dictionary_en_82_765.txt"
        # )

        # self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    def remove_stop_words(self, chat_df: pd.DataFrame, columns=["content"]):
        """
        Removes stopwords from the text.
        :param text: Input text.
        :param stopwords: List of stopwords to remove.
        :return: Text without stopwords.
        """
        for column in columns:
            chat_df[column] = chat_df[column].progress_apply(
                lambda x: " ".join(
                    [word for word in x.split() if word not in STOP_WORDS]
                )
            )
        return chat_df

    def __apply_correct_text(
        self, text, confidence_threshold=0.9, max_edit_distance=1, min_word_length=3
    ):
        if len(text.strip()) <= 1:
            return text
        words = text.split()
        corrected_words = []

        for word in words:
            if len(word) <= min_word_length:
                corrected_words.append(word)
                continue
            suggestions = self.sym_spell.lookup(
                word, Verbosity.TOP, max_edit_distance=max_edit_distance
            )

            if suggestions:
                best_suggestion = suggestions[0]
                confidence = best_suggestion.count / sum(s.count for s in suggestions)

                # Apply correction only if confidence is high
                if confidence >= confidence_threshold:
                    corrected_words.append(best_suggestion.term)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def correct_text(
        self,
        chat_df: pd.DataFrame,
        columns=["content"],
        confidence_threshold=0.9,
        max_edit_distance=1,
    ):
        """
        The function corrects text in specified columns of a DataFrame using a spell-checking algorithm
        with given confidence threshold and maximum edit distance.
        """
        for column in columns:
            chat_df[column] = chat_df[column].progress_apply(
                lambda x: self.__apply_correct_text(
                    x,
                    confidence_threshold=confidence_threshold,
                    max_edit_distance=max_edit_distance,
                )
            )
        return chat_df

    def get_only_text_rows(
        self, chat_df: pd.DataFrame, columns=["content"]
    ) -> pd.DataFrame:
        """
        Extracts only text from the chat data.
        :param chat_df: Input chat data.
        :return: Chat data with only text.
        """
        if "type" in chat_df.columns:
            chat_df = chat_df[chat_df["type"] == "text"]
            return chat_df

        return chat_df[chat_df[columns].notnull().all(axis=1)]

    def clean_text(
        self, chat_df: pd.DataFrame, text_columns=["content"]
    ) -> pd.DataFrame:
        """
        Cleans and corrects the text.
        :param text: Input text.
        :return: Cleaned and corrected chat data.
        """
        chat_df = self.get_only_text_rows(chat_df, columns=text_columns)
        chat_df = self.correct_text(chat_df, columns=text_columns)
        chat_df = self.remove_stop_words(chat_df, columns=text_columns)
        return chat_df
