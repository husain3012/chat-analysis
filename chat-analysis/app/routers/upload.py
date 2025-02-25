from fastapi import APIRouter, UploadFile, File, Depends
import shutil
from app.config import UPLOAD_FOLDER
from app.database import SessionLocal
from app.models import ChatFile
from sqlalchemy.orm import Session
from datetime import datetime

router = APIRouter(prefix="/upload", tags=["Upload"])


def save_file(file: UploadFile):
    file_path = f"{UPLOAD_FOLDER}{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path


@router.post("/")
async def upload_chat_file(
    file: UploadFile = File(...), db: Session = Depends(SessionLocal)
):
    file_path = save_file(file)
    db_file = ChatFile(filename=file.filename, uploaded_at=datetime.utcnow())
    db.add(db_file)
    db.commit()
    return {"filename": file.filename, "message": "File uploaded successfully"}
