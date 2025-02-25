import logging
import os
from app.config import UPLOAD_FOLDER, PROCESSED_FOLDER, TMP_FOLDER, LOG_FOLDER
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import zipfile
import shutil

# create a logger instance for the application, which logs messages to the console and a file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(
    LOG_FOLDER / f"CHAT_ANALYSIS_{datetime.now().strftime('%Y%m%d')}.log"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())


def save_intermediate_df(filename: str, df):
    filename = filename + ".csv"
    file_path = TMP_FOLDER / filename
    df.to_csv(file_path, index=False)
    return file_path


def save_intermediate_dict(filename: str, data: dict):
    filename = filename + ".json"
    file_path = TMP_FOLDER / filename
    with open(file_path, "w") as f:
        json.dump(data, f, sort_keys=True, default=str)
    return file_path


def load_intermediate_df(filename: str):
    file_path = TMP_FOLDER / filename
    return pd.read_csv(file_path)


def load_intermediate_dict(filename: str):
    file_path = TMP_FOLDER / filename
    with open(file_path, "r") as f:
        return json.load(f)


def extract_chat_from_whatsapp_zip(zip_path: Path) -> Path:
    """
    extract zip folder from whatsapp
    get the chat file
    make a copy of the chat file in tmp folder
    rename the chat file to zip file name
    return the chat file path
    :param zipfile_path:
    :return: chat file path
    """
    whatsapp_chat_dir = UPLOAD_FOLDER / "whatsapp_chat"
    if not os.path.exists(whatsapp_chat_dir):
        os.makedirs(whatsapp_chat_dir)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # delete the existing folder if it exists, even if its not empty
        if os.path.exists(TMP_FOLDER / zip_path.stem):
            shutil.rmtree(TMP_FOLDER / zip_path.stem)

        zip_ref.extractall(TMP_FOLDER / zip_path.stem)
        # by default, the chat file is named '_chat.txt'
        chat_file_path = Path(TMP_FOLDER / zip_path.stem / "_chat.txt")
        # rename the chat file to the zip file name
        new_chat_file_path = " ".join(zip_path.stem.split()[3:]) + ".txt"
        new_chat_file_path = whatsapp_chat_dir / new_chat_file_path
        # remove the file if it exists
        if os.path.exists(new_chat_file_path):
            os.remove(new_chat_file_path)
        os.rename(chat_file_path, new_chat_file_path)
        # remove the extracted folder
        shutil.rmtree(TMP_FOLDER / zip_path.stem)

        return new_chat_file_path
    return None


def save_full_report(report_dict, report_name):
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)
    report_path = PROCESSED_FOLDER / f"{report_name}.json"
    # remove the file if it exists
    if os.path.exists(report_path):
        os.remove(report_path)
    with open(report_path, "w") as f:
        json.dump(report_dict, f, sort_keys=True, default=str)
    return report_path
