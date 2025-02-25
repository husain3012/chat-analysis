from app.services.utils import (
    logger,
    save_intermediate_df,
    save_intermediate_dict,
    extract_chat_from_whatsapp_zip,
    save_full_report,
)
from app.services.parsers import WhatsappParser
from app.services.preprocessing import ChatPreprocessor, CleanTextForNLP
from app.services.context_creation import ContextCreator
from app.services.statistics import ChatStatistics, CallStatistics, AnalysisStatistics
from app.services.ai import LanguageDetection, CompleteAnalysis
from app.database import SessionLocal
from app.models import ProcessingStatus
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path


def cli_pipeline(
    file_path: str, deep: bool = True, sample_size: int = None, batch_size=32
):
    file_path = Path(file_path)
    db = SessionLocal()
    # Step 0. Extract zip if file is a zip
    if file_path.suffix == ".zip":
        logger.info(f"Extracting zip file: {file_path}")
        file_path = extract_chat_from_whatsapp_zip(file_path)
        logger.info(f"Extracted chat file: {file_path}")

    # Step 1. Parse the chat file
    logger.info(f"Parsing chat file: {file_path}")
    whatsapp_parser = WhatsappParser()
    messages_df, calls_df, participants = whatsapp_parser.parse_whatsapp_chat(file_path)
    if sample_size and len(messages_df) > sample_size:
        # take a subset of the messages for analysis, sample_size consecutive messages, starting from the random index within the range of the messages
        start_index = np.random.randint(0, len(messages_df) - sample_size)
        messages_df = messages_df.iloc[start_index : start_index + sample_size]
        calls_df = calls_df.sample(n=min(sample_size, len(calls_df)))

        messages_df = messages_df.sort_values(by=["date", "time"]).reset_index(
            drop=True
        )
        calls_df = calls_df.sort_values(by=["date", "time"]).reset_index(drop=True)

    logger.info(f"Parsed {len(messages_df)} messages from the chat file.")

    # Step 2. Preprocess the chat data
    logger.info("Preprocessing chat data.")
    chat_preprocessor = ChatPreprocessor()
    chat_cleaner = CleanTextForNLP()
    context_creator = ContextCreator(participants=participants)

    raw_text_messages_df = chat_preprocessor.process(
        messages_df, text_columns=["content"]
    )

    clean_text_messages_df = chat_cleaner.clean_text(
        raw_text_messages_df, text_columns=["content"]
    )

    context_5min_clean_raw_df = context_creator.merge(
        raw_text_messages_df, "content", "merged_content", 5
    )
    context_60min_clean_chat_df = context_creator.merge(
        clean_text_messages_df, "content", "merged_content", 60
    )
    context_1day_clean_chat_df = context_creator.merge(
        clean_text_messages_df, "content", "merged_content", 1440
    )
    save_intermediate_df("raw_text_messages_df", raw_text_messages_df)
    save_intermediate_df("clean_text_messages_df", clean_text_messages_df)
    save_intermediate_df("context_5min_clean_raw_df", context_5min_clean_raw_df)
    save_intermediate_df("context_60min_clean_chat_df", context_60min_clean_chat_df)
    save_intermediate_df("context_1day_clean_chat_df", context_1day_clean_chat_df)
    logger.info(f"Preprocessed {len(messages_df)} messages.")
    # Step 3. AI STUFF
    if deep:
        logger.info("Performing AI analysis.")
        language_detection = LanguageDetection()
        clean_text_messages_df_language = language_detection.detect_language_df(
            clean_text_messages_df, text_column="content", batch_size=batch_size
        )

        save_intermediate_df(
            "clean_text_messages_df_language", clean_text_messages_df_language
        )
        complete_analysis = CompleteAnalysis()
        analysis_df = complete_analysis.analyze(
            context_60min_clean_chat_df,
            text_column="merged_content",
            batch_size=batch_size,
        )
        save_intermediate_df("analysis_df", analysis_df)

        analysis = AnalysisStatistics(
            analysis_df, text_column="merged_content", participants=participants
        )
        deep_analysis = analysis.analyze()
        save_intermediate_dict("analysis_stats", deep_analysis)
        logger.info("AI analysis completed.")

    # skip for now
    # Step 4. Statistics
    logger.info("Calculating chat statistics.")
    chat_analyzer = ChatStatistics(
        chat_df=raw_text_messages_df, participants=participants
    )
    call_analyzer = CallStatistics(calls_df=calls_df)
    chat_stats = chat_analyzer.analyze()
    call_stats = call_analyzer.analyze()
    save_intermediate_dict("chat_stats", chat_stats)
    save_intermediate_dict("call_stats", call_stats)
    logger.info("Chat statistics calculated.")

    results = {
        "participants": participants,
        "chat_stats": {"overview": chat_stats, "deep_analysis": deep_analysis},
        "call_stats": call_stats,
    }

    report_path = save_full_report(results, file_path.stem)

    logger.info(f"Full report saved at: {report_path}")

    return report_path
