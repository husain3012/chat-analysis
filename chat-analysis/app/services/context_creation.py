import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


class ContextCreator:
    def __init__(self, participants):
        """
        Creates contextual messages by merging consecutive messages within a specified time threshold,
        even if another sender interrupts the flow.

        :param participants: List of all senders and receivers in the chat.
        """
        self.participants = participants

    def merge(
        self,
        df: pd.DataFrame,
        text_column: str,
        merged_text_column: str,
        time_threshold: int,
    ) -> pd.DataFrame:
        """
        Merge consecutive messages within a time threshold, maintaining context even if another sender interrupts.

        :param df: Pandas DataFrame with sender, receiver, content, date, and time.
        :param text_column: The column containing text content.
        :param merged_text_column: The column name for merged text output.
        :param time_threshold: Time threshold (in seconds) to consider messages as part of the same context.

        :return: Merged DataFrame with context-based messages, sorted by date and time.
        """
        df = df[df[text_column].notnull()]
        df = df.sort_values(by=["date", "time"]).reset_index(drop=True)

        merged_data = []
        ongoing_contexts = {
            p: {"messages": [], "start": None, "end": None} for p in self.participants
        }

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Messages"):
            sender, receiver, content, date, time = (
                row["sender"],
                row["receiver"],
                row[text_column],
                row["date"],
                row["time"],
            )
            timestamp = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")

            for p in self.participants:
                if ongoing_contexts[p]["messages"]:
                    last_timestamp = ongoing_contexts[p]["start"]

                    # Check if the current message falls within the time threshold
                    if (timestamp - last_timestamp) <= timedelta(
                        minutes=time_threshold
                    ):
                        ongoing_contexts[p]["messages"].append(content)
                        ongoing_contexts[p]["end"] = timestamp
                    else:
                        # Save completed context for sender `p`
                        merged_data.append(
                            {
                                "sender": p,
                                "receiver": receiver,
                                f"{merged_text_column}": " ".join(
                                    ongoing_contexts[p]["messages"]
                                ),
                                "date": ongoing_contexts[p]["start"].date(),
                                "start_datetime": ongoing_contexts[p]["start"].strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "end_datetime": last_timestamp.strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            }
                        )
                        # Reset context after saving
                        ongoing_contexts[p] = {
                            "messages": [],
                            "start": None,
                            "end": None,
                        }

            # Start a new context for the current sender
            ongoing_contexts[sender] = {
                "messages": ongoing_contexts[sender]["messages"] + [content],
                "start": timestamp,
                "end": timestamp,
            }

        # Append any remaining messages
        for p in self.participants:
            if ongoing_contexts[p]["messages"]:
                merged_data.append(
                    {
                        "sender": p,
                        "receiver": receiver,
                        f"{merged_text_column}": " ".join(
                            ongoing_contexts[p]["messages"]
                        ),
                        "date": ongoing_contexts[p]["start"].date(),
                        "start_datetime": ongoing_contexts[p]["start"].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "end_datetime": ongoing_contexts[p]["end"].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                )

        merged_data = pd.DataFrame(merged_data)
        merged_data = merged_data[
            merged_data[merged_text_column].str.strip().str.len() > 3
        ]

        # Ensure final sorting by date and time
        merged_data["start_datetime"] = pd.to_datetime(merged_data["start_datetime"])
        merged_data = merged_data.sort_values(by=["start_datetime"]).reset_index(
            drop=True
        )

        return merged_data
