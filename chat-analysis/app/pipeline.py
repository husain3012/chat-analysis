from app.services.utils import (
    get_logger,
    save_intermediate_df,
    save_intermediate_dict,
    extract_chat_from_whatsapp_zip,
    save_full_report,
    make_folders,
)
from app.services.parsers import WhatsappParserAndroid, WhatsappParserIOS
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
    platform, file_path: str, deep: bool = True, sample_size: int = None, batch_size=32
):
    make_folders()
    logger = get_logger()
    logger.info(f"Processing for platform: {platform}")

    file_path = Path(file_path)
    db = SessionLocal()

    if file_path.suffix == ".zip":
        logger.info(f"Extracting zip file: {file_path}")
        file_path = extract_chat_from_whatsapp_zip(file_path)
        logger.info(f"Extracted chat file: {file_path}")

    # Step 1. Parse the chat file
    logger.info(f"Parsing chat file: {file_path}")
    whatsapp_parser = None

    if str(platform) == "android":
        whatsapp_parser = WhatsappParserAndroid()
    elif str(platform) == "ios":
        whatsapp_parser = WhatsappParserIOS()

    messages_df, calls_df, participants = whatsapp_parser.parse_whatsapp_chat(file_path)

    if sample_size and len(messages_df) > sample_size:
        messages_df = messages_df.tail(sample_size)
        calls_df = calls_df.tail(sample_size)

    logger.info(f"Parsed {len(messages_df)} messages from the chat file.")

    chat_stats = ChatStatistics(
        chat_df=messages_df, participants=participants
    ).analyze()
    
    preprocessed_chat_df = ChatPreprocessor().process(messages_df)

    cleaned_chat_df = CleanTextForNLP().clean_text(preprocessed_chat_df)
    


    chat_stats = ChatStatistics(
        chat_df=preprocessed_chat_df, participants=participants
    ).analyze() 

    logger.info("Basic chat statistics:")
    chat_analyzer = ChatStatistics(
        chat_df=messages_df, participants=participants
    )
    chat_stats = chat_analyzer.analyze()
  

   
    logger.info(f"Preprocessed {len(messages_df)} messages.")
    # Step 3. AI STUFF
    deep_analysis = None
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
    
    call_analyzer = CallStatistics(calls_df=calls_df, participants=participants)
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
