import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/sqlite.db")
UPLOAD_FOLDER = "data/uploads/"
TMP_FOLDER = "data/tmp/"
PROCESSED_FOLDER = "data/processed/"
LOG_FOLDER = "logs/"
DATA_PATH = "data/"


# Path
UPLOAD_FOLDER = Path(UPLOAD_FOLDER)
TMP_FOLDER = Path(TMP_FOLDER)
PROCESSED_FOLDER = Path(PROCESSED_FOLDER)
DATA_PATH = Path(DATA_PATH)
LOG_FOLDER = Path(LOG_FOLDER)


# Settings
BATCH_SIZE = 32
LANGUAGE_DETECTION = True
DEFAULT_CHAT_LANGUAGE = "en"
LANGUAGE_TRANSLATION = True
