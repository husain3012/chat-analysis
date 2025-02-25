import pandas as pd
import re
from collections import Counter
from datetime import datetime, timedelta
import emoji
import numpy as np
from app.config import DATA_PATH
import json
from tqdm import tqdm


class ChatStatistics:
    def __init__(self, chat_df, participants):
        """
        Initialize with chat dataframe and list of participants.
        :param chat_df: DataFrame containing chat history.
        :param participants: List of participants to analyze.
        """
        self.df = chat_df.copy()
        self.participants = participants
        self.df["datetime"] = pd.to_datetime(
            self.df["date"].astype(str) + " " + self.df["time"].astype(str)
        )
        self.df = self.df.sort_values(by=["datetime"]).reset_index(drop=True)
        wordlist_json = DATA_PATH / "static/wordlists.json"
        self.compliment_wordlist = []
        self.apology_wordlist = []
        self.swearing_wordlist = []
        self.question_wordlist = []

        with open(wordlist_json, "r") as f:
            wordlist = json.load(f)
            self.compliment_wordlist = wordlist["compliment"]
            self.apology_wordlist = wordlist["apology"]
            self.question_wordlist = wordlist["questioning"]
            self.swearing_wordlist = wordlist["swearing"]

    def most_used_words_phrases(self, top_n=10, min_len=3):
        """
        Find the most used words and phrases.
        :param top_n: Number of top words/phrases to return.
        :return: Dictionary of most used words and phrases.
        """
        words = []
        for content in tqdm(
            self.df["content"],
            desc="Calculating Word Frequencies",
            total=self.df.shape[0],
        ):
            if content:
                words.extend(content.lower().split())

        words = [
            word for word in words if len(word) >= min_len and not emoji.is_emoji(word)
        ]
        # remove special characters
        words = [re.sub(r"[^A-Za-z0-9]+", "", word) for word in words]

        word_counts = Counter(words).most_common(top_n)
        return dict(word_counts)

    def who_texts_first_most(self):
        """
        Determine who texts first most often in a day.
        :return: Dictionary with counts of who initiated chats.
        """
        first_senders = self.df.groupby("date")["sender"].first().value_counts()
        return first_senders.to_dict()

    def who_sends_longer_messages(self):
        """
        Determine who sends longer messages on average.
        :return: Dictionary of average message length per sender.
        """
        self.df["msg_length"] = (
            self.df["content"].dropna().apply(lambda x: len(x.split()))
        )
        avg_length = self.df.groupby("sender")["msg_length"].mean()
        return avg_length.to_dict()

    def longest_message(self):
        """
        Details of the longest message by each participant.
        :return: Dictionary of longest message details per sender.
        """
        if "msg_length" not in self.df.columns:
            self.df["msg_length"] = (
                self.df["content"].dropna().apply(lambda x: len(x.split()))
            )

        longest_messages = {}
        for sender in self.participants:
            sender_df = self.df[self.df["sender"] == sender]
            longest_msg_idx = sender_df["msg_length"].idxmax()
            longest_messages[sender] = {
                "content": sender_df.loc[longest_msg_idx, "content"],
                "length": sender_df.loc[longest_msg_idx, "msg_length"],
                "datetime": sender_df.loc[longest_msg_idx, "datetime"],
            }
        return longest_messages

    def average_reply_time(self, time_threshold=8 * 60):
        """
        Calculate average reply times for each participant, ignoring long gaps.
        :param time_threshold: Maximum gap (in minutes) to consider as a valid reply.
        :return: Dictionary of average reply times in minutes.
        """
        reply_times = {p: [] for p in self.participants}
        prev_sender = None
        prev_time = None

        for _, row in self.df.iterrows():
            if prev_sender and prev_sender != row["sender"]:
                diff = (row["datetime"] - prev_time).total_seconds() / 60  # in minutes

                # Ignore long gaps (e.g., overnight breaks)
                if diff <= time_threshold:
                    reply_times[prev_sender].append(diff)

            prev_sender = row["sender"]
            prev_time = row["datetime"]

        avg_times = {
            p: np.mean(reply_times[p]) if reply_times[p] else None
            for p in self.participants
        }
        return avg_times

    def who_apologizes_more(self):
        """
        Determine who apologizes more often.
        :return: Dictionary of apology counts per sender.
        """

        apology_counts = {p: 0 for p in self.participants}

        for _, row in self.df.iterrows():
            if row["content"]:
                if any(
                    word in row["content"].lower() for word in self.apology_wordlist
                ):
                    apology_counts[row["sender"]] += 1

        return apology_counts

    def who_compliments_more(self):
        """
        Determine who gives more compliments.
        :return: Dictionary of compliment counts per sender.
        """

        compliment_counts = {p: 0 for p in self.participants}

        for _, row in self.df.iterrows():
            if row["content"]:
                if any(
                    word in row["content"].lower() for word in self.compliment_wordlist
                ):
                    compliment_counts[row["sender"]] += 1

        return compliment_counts

    def who_asks_more_questions(self):
        """
        Determine who asks more questions.
        :return: Dictionary of question counts per sender.
        """
        question_counts = {p: 0 for p in self.participants}

        for _, row in self.df.iterrows():
            if row["content"]:
                if any(
                    word in row["content"].lower() for word in self.question_wordlist
                ):
                    question_counts[row["sender"]] += 1

        return question_counts

    def who_swears_more(self):
        """
        Determine who swears more often.
        :return: Dictionary of swearing counts per sender.
        """
        swear_counts = {p: 0 for p in self.participants}

        for _, row in self.df.iterrows():
            if row["content"]:
                if any(
                    word in row["content"].lower() for word in self.swearing_wordlist
                ):
                    swear_counts[row["sender"]] += 1

        return swear_counts

    def most_active_chat_times(self):
        """
        Find the most active chat times.
        :return: Dictionary of active hours with message counts.
        """
        self.df["hour"] = self.df["datetime"].dt.hour
        active_hours = self.df["hour"].value_counts().sort_index()
        return active_hours.to_dict()

    def most_used_emojis(self, top_n=10):
        """
        Find the most used emojis overall and per participant.
        :param top_n: Number of top emojis to return.
        :return: Tuple (overall top emojis, participant-wise emoji usage).
        """
        emoji_counts = Counter()
        participant_emoji_counts = {p: Counter() for p in self.participants}

        for _, row in self.df.iterrows():
            emojized_text = emoji.emojize(row["content"]) if row["content"] else ""
            emojis_in_text = emoji.emoji_list(emojized_text) if emojized_text else []
            emoji_list = [e["emoji"] for e in emojis_in_text]

            if emoji_list:
                emoji_counts.update(emoji_list)
                participant_emoji_counts[row["sender"]].update(emoji_list)

        top_overall = emoji_counts.most_common(top_n)
        top_participant_wise = {
            p: participant_emoji_counts[p].most_common(top_n) for p in self.participants
        }

        return dict(top_overall), {k: dict(v) for k, v in top_participant_wise.items()}

    def media_stats(self):
        """
        Get number of media messages sent by each participant, along with media types.
        """
        media_messages = self.df[self.df["type"] != "text"]
        unique_media_types = media_messages["type"].unique()
        media_stats = {p: {t: 0 for t in unique_media_types} for p in self.participants}
        for p in self.participants:
            participant_media = media_messages[media_messages["sender"] == p]
            media_stats[p] = participant_media["type"].value_counts().to_dict()
        return media_stats

    def analyze(self):
        """
        Analyze chat data.
        :return: Dictionary of analysis results.
        """
        results = {
            "most_used_words_phrases": self.most_used_words_phrases(),
            "who_texts_first_most": self.who_texts_first_most(),
            "who_sends_longer_messages": self.who_sends_longer_messages(),
            "longest_message": self.longest_message(),
            "average_reply_time": self.average_reply_time(),
            "who_apologizes_more": self.who_apologizes_more(),
            "media_stats": self.media_stats(),
            "who_asks_more_questions": self.who_asks_more_questions(),
            "who_swears_more": self.who_swears_more(),
            "who_compliments_more": self.who_compliments_more(),
            "most_active_chat_times": self.most_active_chat_times(),
            "most_used_emojis": self.most_used_emojis(),
        }

        return results
