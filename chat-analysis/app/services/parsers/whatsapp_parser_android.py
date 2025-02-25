import re
import pandas as pd
from tqdm import tqdm
from datetime import datetime


class WhatsappParserAndroid:
    def __init__(self):

        self.media_regex = r"\<media omitted\>$"
        self.deleted_by_me = r"you deleted this message\.$"
        self.deleted_by_others = r"this message was deleted\.$"

        self.edited_message = r"(.+) \<this message was edited\>$"

        self.pattern = r"(\d{2}\/\d{2}\/\d{2}),\s(\d{1,2}\:\d{1,2}.[ap]m) - (.*?): (.*)"

        self.duration_multiplier = {"sec": 1, "min": 60, "h": 3600}

        self.WHATSAPP_DEFAULT_MESSAGES = [
            "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them."
        ]

    def get_participants(self, chat_file):

        participants = set()
        with open(chat_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                match = re.match(self.pattern, line.strip())
                if match:
                    participants.add(match.group(3))
        return list(participants)

    def __extract_and_remove_links(self, content):
        link_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        link = re.search(link_regex, content)
        if link:
            content = content.replace(link.group(), "")
        return content, link.group() if link else None

    def __get_text_info(self, content):
        text_type = "unknown"
        text_content = content
        if re.search(self.media_regex, content):
            text_type = "media"
            text_content = None
            return text_type, text_content
        return text_type, text_content

    def parse_whatsapp_chat(
        self, chat_file, participants=None
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Parse a WhatsApp chat file and return the messages and calls as DataFrames.
        :param chat_file: Path to the chat file
        :param participants: List of participants in the chat, if not provided, it will be extracted from the chat file
        :return: Tuple of DataFrames containing messages and calls, and a list of participants
        """
        if participants is None:
            participants = self.get_participants(chat_file)
        messages, calls = [], []

        number_of_lines = sum(1 for line in open(chat_file, "r", encoding="utf-8"))
        first_person = participants[0]
        second_person = participants[1]

        with open(chat_file, "r", encoding="utf-8") as file:
            for line in tqdm(
                file, desc="Parsing chat file", unit="line", total=number_of_lines
            ):
                if any(
                    [
                        line.strip().lower() == message.strip().lower()
                        for message in self.WHATSAPP_DEFAULT_MESSAGES
                    ]
                ):
                    continue

                match = re.match(self.pattern, line.strip())
                if not match:
                    continue

                date, time, sender, content = match.groups()

                date = datetime.strptime(date, "%d/%m/%y").date()
                time = datetime.strptime(time, "%I:%M %p").time()
                # add 00 seconds to the time
                time = time.replace(second=0)

                content = content.strip().lower()

                if len(participants) > 2:
                    receiver = "group"
                else:
                    receiver = second_person if sender == first_person else first_person

                if any(
                    [
                        re.search(self.media_regex, content),
                    ]
                ):
                    text_type, text_content = self.__get_text_info(content)
                    text_content = (
                        self.__extract_and_remove_links(text_content)[0]
                        if text_content
                        else None
                    )
                    messages.append(
                        [sender, receiver, text_content, date, time, text_type]
                    )
                elif content.strip().lower() != "":
                    messages.append([sender, receiver, content, date, time, "text"])
                else:
                    messages.append([sender, receiver, None, date, time, "unknown"])
            # Save to CSV
        messages = pd.DataFrame(
            messages, columns=["sender", "receiver", "content", "date", "time", "type"]
        )

        messages.astype

        dummy_calls_df = pd.DataFrame(
            columns=[
                "sender",
                "receiver",
                "duration",
                "type",
                "group_call_invites",
                "silence_reason",
                "date",
                "time",
            ]
        )

        return messages, dummy_calls_df, participants
