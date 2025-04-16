import pandas as pd
import emoji
import app.config as cfg
import json
from symspellpy import SymSpell, Verbosity
import pkg_resources
import nltk
import re
from tqdm import tqdm
from spello.model import SpellCorrectionModel  
from app.services.ai import LanguageDetection, LanguageTranslation

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
        abbreviations_json = cfg.DATA_PATH / "static/abbreviations.json"
        self.abbreviations = {}

        with open(abbreviations_json, "r") as f:
            self.abbreviations = json.load(f)
            self.abbreviations = {
                k.lower(): v.lower() for k, v in self.abbreviations.items()
            }

    def demojize(self, chat_df: pd.DataFrame, column):
        """
        Convert emojis their text description
        :param columns: list of columns to process
        """
        tqdm.pandas(desc="Demojizing")
        chat_df[column] = chat_df[column].progress_apply(
            lambda x: (emoji.demojize(x) if type(x) == str else x)
        )
        return chat_df

    def expand_abbreviations(self, chat_df: pd.DataFrame, column):
        """
        Expands abbreviations in the chat data
        :param abbreviations: dictionary with abbreviations as keys and their expansions as values
        """
        tqdm.pandas(desc="Expanding Abbreviations")
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

    def process(self, chat_df: pd.DataFrame, text_column = "content"):
        """
        Process chat data, end to end.
        Steps:
        1. Expand abbreviations
        2. Convert emojis to text

        """
        chat_df = self.expand_abbreviations(chat_df, column=text_column)
        chat_df = self.demojize(chat_df, column=text_column)
        return chat_df


class CleanTextForNLP:
    def __init__(self):
        """
        Cleans and corrects text using spell correction and remove special characters.
        :param dictionary_path: Path to a frequency-based dictionary file.
        :param max_edit_distance: Maximum allowed edit distance for suggestions.
        """
        self.spello = SpellCorrectionModel(language="en")
        self.spello.load(cfg.DATA_PATH/'static/model.pkl')
    def remove_stop_words(self, chat_df: pd.DataFrame, column="content"):
        """
        Removes stopwords from the text.
        :param text: Input text.
        :param stopwords: List of stopwords to remove.
        :return: Text without stopwords.
        """
        chat_df[column] = chat_df[column].progress_apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in STOP_WORDS]
            )
        )
        return chat_df

    def __apply_correct_text(
        self, text
    ):
        if len(text.strip()) <= 1:
            return text 
        return self.spello.spell_correct(text)['spell_corrected_text']

    def correct_text(
        self,
        chat_df: pd.DataFrame,
        column = "content",
    ):
        """
        The function corrects text in specified columns of a DataFrame using a spell-checking algorithm
        with given confidence threshold and maximum edit distance.
        """
        if column not in chat_df.columns:
            raise ValueError(f"Column {column} not found in the chat data.")
        if "language" not in chat_df.columns:
            raise ValueError("Language column not found in the chat data.")
       
       
        chat_df[column] = chat_df.progress_apply(
            lambda x: self.__apply_correct_text(x[column]) if x["language"] == "en" else x[column],
            axis=1
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

        language_detector = LanguageDetection()
        language_translator = LanguageTranslation()
        if cfg.LANGUAGE_DETECTION:
            chat_df = language_detector.detect_language_df(
                chat_df, text_column="content", batch_size=cfg.BATCH_SIZE
            )
        else:
            chat_df["language"] = cfg.DEFAULT_CHAT_LANGUAGE

        chat_df = self.correct_text(chat_df, column="content")
       
        if cfg.LANGUAGE_TRANSLATION:
            chat_df = language_translator.translate_language_df(chat_df)
        else:
            chat_df["translated"] = chat_df["content"]

        chat_df = self.remove_stop_words(chat_df, columns="translated")
        
        return chat_df
