from .emotion_analysis import EmotionAnalysis
from .language_detection import LanguageDetection
from .sentiment_analysis import SentimentAnalysis
from .toxicity_analysis import ToxicityAnalysis
import pandas as pd


class CompleteAnalysis:
    def __init__(self):
        pass

    def analyze(
        self,
        clean_chat_df: pd.DataFrame,
        text_column: str = "content",
        batch_size: int = 32,
    ) -> pd.DataFrame:

        sentiment_analyzer = SentimentAnalysis()
        emotion_analyzer = EmotionAnalysis()
        toxicity_analyzer = ToxicityAnalysis()

        clean_chat_df["sentiment"] = sentiment_analyzer.analyze_batch(
            clean_chat_df[text_column], batch_size=batch_size
        )
        clean_chat_df["emotion"] = emotion_analyzer.analyze_batch(
            clean_chat_df[text_column], batch_size=batch_size
        )
        clean_chat_df["toxicity"] = toxicity_analyzer.analyze_batch(
            clean_chat_df[text_column], batch_size=batch_size
        )
        return clean_chat_df
