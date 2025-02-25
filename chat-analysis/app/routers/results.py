from fastapi import APIRouter
import pandas as pd
from app.config import PROCESSED_FOLDER

router = APIRouter(prefix="/results", tags=["Results"])


@router.get("/{filename}")
async def get_results(filename: str):
    file_path = f"{PROCESSED_FOLDER}{filename}.csv"
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    except FileNotFoundError:
        return {"error": "Processed file not found"}
