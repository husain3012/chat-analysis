import pandas as pd
import numpy as np
from ast import literal_eval


class AnalysisStatistics:
    SENTIMENT_MAPPING = {
        "Very Negative": -2,
        "Negative": -1,
        "Neutral": 0,
        "Positive": 1,
        "Very Positive": 2,
    }

    def __init__(self, df, text_column, participants):
        """
        Initializes the analyzer with a pre-analyzed DataFrame.
        :param df: DataFrame containing ['sender', 'receiver', 'text_column', 'date', 'sentiment', 'emotion', 'toxicity']
        :param text_column: Column containing text messages.
        :param participants: List of participants in the chat.
        """
        self.df = df.copy()
        self.text_column = text_column
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["week"] = pd.to_datetime(self.df["date"]).dt.to_period(
            "W"
        )  
        
        self.participants = participants
        self._map_sentiment()
        self._parse_toxicity_column()
        self._parse_emotion_column()

    sorted_emotions = [
        "love",
        "caring",
        "gratitude",
        "admiration",
        "approval",
        "pride",
        "joy",
        "amusement",
        "excitement",
        "optimism",
        "relief",
        "desire",
        "surprise",
        "realization",
        "curiosity",
        "confusion",
        "nervousness",
        "fear",
        "sadness",
        "disappointment",
        "remorse",
        "embarrassment",
        "grief",
        "annoyance",
        "anger",
        "disapproval",
        "disgust",
    ]


    def _parse_toxicity_column(self):
        """Parses 'toxicity' into numerical scores."""
        self.df["toxicity_score"] = self.df["toxicity"].apply(
            lambda x: literal_eval(x)[0] if isinstance(x, str) else 0
        )
        self.df["toxicity_type"] = self.df["toxicity"].apply(
            lambda x: literal_eval(x)[1] if isinstance(x, str) else None
        )

    def _parse_emotion_column(self):
        """Extracts dominant emotion for each message."""
        # create a string csv of emotions, with emotions greater than 0.001
        self.df["emotions_str"] = self.df["emotion"].apply(
            lambda x: ", ".join(
                [
                    emotion
                    for emotion, score in (
                        literal_eval(x).items() if type(x) == str else x.items()
                    )
                    if score > 0.001
                ]
            )
        )

    def get_mood_trends(self, rolling_window=7):
        """
        return sentiment and toxicity trends over time in a list of dictionaries
        """
        df_grouped = self.df.groupby("date").mean(numeric_only=True)
        df_grouped["sentiment_rolling"] = (
            df_grouped["sentiment_numeric"]
            .rolling(rolling_window, min_periods=1)
            .mean()
        )
        df_grouped["toxicity_rolling"] = (
            df_grouped["toxicity_score"].rolling(rolling_window, min_periods=1).mean()
        )
        mood_trends = df_grouped[["sentiment_rolling", "toxicity_rolling"]].to_dict(
            orient="records"
        )
        return mood_trends

    def get_emotion_trend(self, ignore_neutral=True):
        """
        return emotion trends over time in a list of dictionaries
        """
        emotion_df = self.df.copy()
        emotion_df = emotion_df[emotion_df["emotion"].notnull()]
        emotion_df["emotion"] = emotion_df["emotion"].apply(
            lambda x: literal_eval(x) if type(x) == str else x
        )

        emotions = set()
        for emotion_dict in emotion_df["emotion"]:
            emotions.update(emotion_dict.keys())  # Extract emotion labels

        if ignore_neutral:
            emotions.discard("neutral")

        # Create columns for each emotion and extract scores
        for emotion in emotions:
            emotion_df[emotion] = emotion_df["emotion"].apply(
                lambda x: x.get(emotion, 0)
            )  # Extract scores

        # Group by week and sum up emotion values
        emotion_df = emotion_df.groupby("week")[list(emotions)].sum()

        # Normalize by total per week
        emotion_df = emotion_df.div(emotion_df.sum(axis=1), axis=0)

        return emotion_df.to_dict(orient="records")

    def get_mood_toxicity_heatmap(self):
        """
        returns a heatmap showing mood & toxicity variation over weeks in form of a list of dictionaries
        """
        df_heatmap = self.df.copy()
        df_heatmap["week"] = df_heatmap["date"].dt.strftime("%Y-%U")
        weekly_avg = df_heatmap.groupby("week")[
            ["sentiment_numeric", "toxicity_score"]
        ].mean()
        weekly_avg["sentiment_numeric"] = pd.cut(
            weekly_avg["sentiment_numeric"],
            bins=5,
            labels=[
                "Very Negative",
                "Negative",
                "Neutral",
                "Positive",
                "Very Positive",
            ],
        )
        weekly_avg["toxicity_score"] = pd.cut(
            weekly_avg["toxicity_score"],
            bins=5,
            labels=["Very Low", "Low", "Moderate", "High", "Very High"],
        )
        weekly_avg = weekly_avg.reset_index()
        heatmap = weekly_avg.to_dict(orient="records")
        return heatmap

    def get_toxicity_summary(self):
        """
        Returns insights on toxicity levels.
        """
        top_toxic_messages = self.df[self.df["toxicity_score"] > 0.5]
        return (
            top_toxic_messages.sort_values(by="toxicity_score", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )



    def love_score(self):
        """
        for each participant, calculates love score by counting number of messages with love emotion
        """
        love_emotions = ["love", "caring"]
        love_score = {}
        for participant in self.participants:
            love_score[participant] = (
                self.df[self.df["sender"] == participant]["emotions_str"]
                .apply(
                    lambda x: sum(
                        1 for emotion in love_emotions if emotion in x.split(", ")
                    )
                )
                .sum()
            )
        # calculate percentage
        total_messages = self.df["sender"].value_counts()
        love_percentage = {
            k: round(v / total_messages[k] * 100, 2) for k, v in love_score.items()
        }

        # if NAN, set to 0
        love_score = {k: 0 if np.isnan(v) else v for k, v in love_score.items()}
        love_percentage = {
            k: 0 if np.isnan(v) else v for k, v in love_percentage.items()
        }

        return {
            "love_score": love_score,
            "love_percentage": love_percentage,
        }

    def hate_score(self):
        """
        for each participant, calculates hate score by counting number of messages with anger and disgust emotions
        """
        hate_emotions = ["anger", "annoyance"]
        hate_score = {}
        for participant in self.participants:
            hate_score[participant] = (
                self.df[self.df["sender"] == participant]["emotions_str"]
                .apply(
                    lambda x: sum(
                        1 for emotion in hate_emotions if emotion in x.split(", ")
                    )
                )
                .sum()
            )
        # calculate percentage
        total_messages = self.df["sender"].value_counts()
        hate_percentage = {
            k: round(v / total_messages[k] * 100, 2) for k, v in hate_score.items()
        }
        # if nan set to 0
        hate_score = {k: 0 if np.isnan(v) else v for k, v in hate_score.items()}
        hate_percentage = {
            k: 0 if np.isnan(v) else v for k, v in hate_percentage.items()
        }
        return {
            "hate_score": hate_score,
            "hate_percentage": hate_percentage,
        }

    def analyze(self):
        """
        Analyzes the chat data and returns a summary.
        """
        summary = {
            "toxicity_summary": self.get_toxicity_summary(),
            "sentiment_summary": self.get_sentiment_summary(),
            "mood_trends": self.get_mood_trends(),
            "emotion_trend": self.get_emotion_trend(),
            "mood_toxicity_heatmap": self.get_mood_toxicity_heatmap(),
            "love_score": self.love_score(),
            "hate_score": self.hate_score(),
        }
        return summary
