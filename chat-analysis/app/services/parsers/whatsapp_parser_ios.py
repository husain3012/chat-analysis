import re
import pandas as pd
from tqdm import tqdm
from datetime import datetime


class WhatsappParserIOS:
    def __init__(self):

        self.NON_TEXT_MARKER = "\u200e"  # Used for media and call logs

        self.MEDIA_TYPES = {
            "image omitted": "image",
            "audio omitted": "audio",
            "video omitted": "video",
            "gif omitted": "gif",
            "sticker omitted": "sticker",
            "contact card omitted": "contact_card",
        }

        self.call_regex = r"(voice call|video call)\, \d (min|sec|h)"
        self.missed_call_regex = r"missed (voice|video) call"
        self.no_answer_call_regex = r"(voice|video) call\, no answer"
        self.failed_call_regex = r"call failed\, try again"
        self.group_call_regex = r"group call\, \d invited"
        self.silenced_call_regex = r"(silenced voice call|silenced video call), (.+)$"
        self.attachment_regex = (
            r"(image|audio|video|gif|sticker|contact card|video note) omitted"
        )
        self.location_regex = r"location: (.+)$"
        self.document_regex = r".* document omitted$"
        self.deleted_by_me = r"you deleted this message\.$"
        self.deleted_by_others = r"this message was deleted\.$"
        self.blocked = r"you blocked this contact$"
        self.unblocked = r"you unblocked this contact$"
        self.edited_message = r"(.+) \<this message was edited\>$"

        self.pattern = (
            r"\[(\d{2}/\d{2}/\d{2}), (\d{1,2}:\d{2}:\d{2} [APM]{2})\] (.*?): (.*)"
        )

        self.duration_multiplier = {"sec": 1, "min": 60, "h": 3600}

        self.CALL_TYPES = {
            "voice call": "voice_call",
            "video call": "video_call",
            "missed voice call": "missed_voice_call",
            "missed video call": "missed_video_call",
            "voice call, no answer": "missed_voice_call",
            "video call, no answer": "missed_video_call",
            "failed call": "failed_call",
            "group call": "group_call",
            "silenced voice call": "silenced_voice_call",
            "silenced video call": "silenced_video_call",
        }
        self.TEXT_TYPES = {
            "location": "location",
            "document": "document",
            "deleted_by_me": "deleted_by_me",
            "deleted_by_others": "deleted_by_others",
            "blocked": "blocked",
            "unblocked": "unblocked",
            "text_edited_message": "text_edited_message",
            "unknown": "unknown",
        }
        self.ATTACHMENT_TYPES = {
            "image": "image",
            "audio": "audio",
            "video": "video",
            "gif": "gif",
            "sticker": "sticker",
            "contact card": "contact_card",
        }

        self.WHATSAPP_DEFAULT_MESSAGES = [
            "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them."
        ]

    def get_participants(self, chat_file):

        person_regex = r"\[.*\] (.+?)\: .*"
        # 28/05/24, 11:52 pm - Sarfraz: Anyway

        participants = set()
        with open(chat_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.replace(self.NON_TEXT_MARKER, "")
                match = re.match(person_regex, line.strip())
                if match:
                    participants.add(match.group(1))
        return list(participants)

    def __extract_and_remove_links(self, content):
        link_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        link = re.search(link_regex, content)
        if link:
            content = content.replace(link.group(), "")
        return content, link.group() if link else None

    def __get_call_info(self, content):

        call_type = "unknown_call"
        duration = 0
        silence_reason = None
        group_call_invites = 0

        if re.search(self.call_regex, content):
            call_type = self.CALL_TYPES.get(
                re.search(r"voice call|video call", content).group(), "unknown_call"
            )
            duration = int(re.search(r"\d+", content).group())
            duration *= self.duration_multiplier[
                re.search(r"(sec|min|h)", content).group()
            ]
            return call_type, duration, silence_reason, group_call_invites

        if re.search(self.missed_call_regex, content):
            call_type = self.CALL_TYPES.get(
                re.search(r"missed voice call|missed video call", content).group(),
                "unknown_call",
            )
            return call_type, duration, silence_reason, group_call_invites
        if re.search(self.no_answer_call_regex, content):
            call_type = self.CALL_TYPES.get(
                re.search(self.no_answer_call_regex, content).group(), "unknown_call"
            )
            return call_type, duration, silence_reason, group_call_invites
        if re.search(self.failed_call_regex, content):
            call_type = self.CALL_TYPES["failed call"]
            return call_type, duration, silence_reason, group_call_invites
        if re.search(self.group_call_regex, content):
            call_type = self.CALL_TYPES.get(
                re.search(r"group call", content).group(), "unknown_call"
            )
            group_call_invites = int(
                re.search(r"\d invited", content).group().split()[0]
            )
            return call_type, duration, silence_reason, group_call_invites
        if re.search(self.silenced_call_regex, content):
            call_type = self.CALL_TYPES.get(
                re.search(r"silenced voice call|silenced video call", content).group(),
                "unknown_call",
            )
            silence_reason = re.search(
                r"silenced voice call|silenced video call, (.+)$", content
            ).group(1)
            return call_type, duration, silence_reason, group_call_invites
        return call_type, duration, silence_reason

    def __get_text_info(self, content):
        text_type = "unknown"
        text_content = content
        if re.search(self.attachment_regex, content):
            text_type = re.search(
                r"image|audio|video|gif|sticker|contact card|video note", content
            ).group()
            text_type = self.ATTACHMENT_TYPES.get(text_type, "unknown")
            text_content = None
            return text_type, text_content
        if re.search(self.location_regex, content):
            text_type = "location"
            text_type = self.TEXT_TYPES.get(text_type, "unknown")
            text_content = re.search(r"location: (.+)$", content).group(1)
            return text_type, text_content
        if re.search(self.document_regex, content):
            text_type = "document"
            text_type = self.TEXT_TYPES.get(text_type, "unknown")
            text_content = re.search(r".* document omitted$", content).group()
            return text_type, text_content
        if re.search(self.deleted_by_me, content):
            text_type = self.TEXT_TYPES.get("deleted_by_me")
            text_content = None
            return text_type, text_content
        if re.search(self.deleted_by_others, content):
            text_type = self.TEXT_TYPES.get("deleted_by_others")
            text_content = None
            return text_type, text_content
        if re.search(self.blocked, content):
            text_type = self.TEXT_TYPES.get("blocked")
            text_content = None
            return text_type, text_content
        if re.search(self.unblocked, content):
            text_type = self.TEXT_TYPES.get("unblocked")
            text_content = None
            return text_type, text_content
        if re.search(self.edited_message, content):
            text_type = self.TEXT_TYPES.get("text_edited_message")
            text_content = re.search(
                r"(.+) \<this message was edited\>$", content
            ).group(1)
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
                match = re.match(
                    self.pattern, line.strip().replace(self.NON_TEXT_MARKER, "")
                )
                if not match:
                    continue

                date, time, sender, content = match.groups()

                date = datetime.strptime(date, "%d/%m/%y").date()
                time = datetime.strptime(time, "%I:%M:%S %p").time()
                content = content.strip().lower()
                if any(
                    [
                        content == message.strip().lower()
                        for message in self.WHATSAPP_DEFAULT_MESSAGES
                    ]
                ):
                    continue
                if len(participants) > 2:
                    receiver = "group"
                else:
                    receiver = second_person if sender == first_person else first_person

                if any(
                    [
                        re.search(self.call_regex, content),
                        re.search(self.missed_call_regex, content),
                        re.search(self.failed_call_regex, content),
                        re.search(self.group_call_regex, content),
                        re.search(self.silenced_call_regex, content),
                        re.search(self.no_answer_call_regex, content),
                    ]
                ):
                    call_type, duration, silence_reason, group_call_invites = (
                        self.__get_call_info(content)
                    )
                    calls.append(
                        [
                            sender,
                            receiver,
                            duration,
                            call_type,
                            group_call_invites,
                            silence_reason,
                            date,
                            time,
                        ]
                    )
                elif any(
                    [
                        re.search(self.attachment_regex, content),
                        re.search(self.location_regex, content),
                        re.search(self.document_regex, content),
                        re.search(self.deleted_by_me, content),
                        re.search(self.deleted_by_others, content),
                        re.search(self.blocked, content),
                        re.search(self.unblocked, content),
                        re.search(self.edited_message, content),
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
                elif self.NON_TEXT_MARKER not in content:
                    messages.append([sender, receiver, content, date, time, "text"])
                else:
                    messages.append([sender, receiver, None, date, time, "unknown"])
            # Save to CSV
        messages = pd.DataFrame(
            messages, columns=["sender", "receiver", "content", "date", "time", "type"]
        )
        calls = pd.DataFrame(
            calls,
            columns=[
                "sender",
                "receiver",
                "duration",
                "type",
                "group_call_invites",
                "silence_reason",
                "date",
                "time",
            ],
        )
        # set datatypes of columns
        messages.astype

        return messages, calls, participants
