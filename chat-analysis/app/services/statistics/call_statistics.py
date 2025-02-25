import pandas as pd
import numpy as np
from collections import Counter


class CallStatistics:
    def __init__(self, calls_df):
        """
        Initializes the CallStatsAnalyzer with a calls dataframe.
        Assumes the dataframe has columns: sender, receiver, duration, type, date, time.
        """
        self.df = calls_df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["time"] = pd.to_datetime(
            self.df["time"], format="%H:%M:%S"
        ).dt.hour  # Extract only hours for analysis

    def get_average_call_duration(self):
        """Returns the average call duration (excluding missed/silenced calls)."""
        valid_calls = self.df[self.df["duration"] > 0]
        return valid_calls["duration"].mean() if not valid_calls.empty else 0

    def get_most_frequent_call_type(self):
        """Returns the most frequently occurring call type."""
        return self.df["type"].mode()[0] if not self.df.empty else None

    def get_average_call_frequency_per_day(self):
        """Returns the average number of calls made per day."""
        return self.df.groupby("date").size().mean()

    def get_days_with_no_calls(self):
        """Returns dates where no calls were made."""
        all_days = pd.date_range(self.df["date"].min(), self.df["date"].max())
        called_days = self.df["date"].unique()
        return set(all_days.date) - set(called_days)

    def get_days_with_most_calls(self, top_n):
        """Returns the day(s) with the most calls."""
        max_calls = self.df["date"].value_counts().head(top_n)
        days_with_most_calls = max_calls.index.strftime("%B %d, %Y")
        # for each day in days_with_most_calls, calculate number of calls made by each participant
        calls_per_day = {}
        for day in days_with_most_calls:
            calls_per_day[day] = (
                self.df[self.df["date"].dt.strftime("%B %d, %Y") == day]["sender"]
                .value_counts()
                .to_dict()
            )
        return max_calls.to_dict(), calls_per_day

    def get_most_active_call_times(self):
        """Returns the most common call hours (peak hours)."""
        return self.df["time"].value_counts().to_dict()

    def get_longest_call_details(self):
        """Returns details of the longest calls made, in a dictionary format."""
        longest_call = self.df[self.df["duration"] == self.df["duration"].max()]
        if not longest_call.empty:
            return longest_call.to_dict(orient="records")[0]
        return {}

    def get_total_calls_per_participant(self):
        """Returns the total calls made per participant."""
        return self.df["sender"].value_counts().to_dict()

    def get_type_of_calls_count(self):
        """Returns the count of each call type."""
        return self.df["type"].value_counts().to_dict()
    
    def get_missed_calls_per_participant(self):
        """Returns the count of missed calls per participant, i.e. who never picks up the phone."""
        return self.df[self.df["type"] == "Missed"]["receiver"].value_counts().to_dict()
    def analyze(self):
        """Analyzes the call data and returns a summary."""
        summary = {
            "average_call_duration": self.get_average_call_duration(),
            "most_frequent_call_type": self.get_most_frequent_call_type(),
            "average_call_frequency_per_day": self.get_average_call_frequency_per_day(),
            "missed_calls_per_participant": self.get_missed_calls_per_participant(),
            "most_active_call_times": self.get_most_active_call_times(),
            "longest_call_details": self.get_longest_call_details(),
            "total_calls_per_participant": self.get_total_calls_per_participant(),
            "type_of_calls_count": self.get_type_of_calls_count(),
        }
        return summary
