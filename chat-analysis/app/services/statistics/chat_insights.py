class ChatInsights:
    def __init__(self, chat_df, participants):
        """
        Initializes the chat insights analyzer.
        :param chat_df: DataFrame with chat messages.
        :param participants: List of participants in the chat.
        """
        self.chat_df = chat_df.copy()
        self.participants = participants
        


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
