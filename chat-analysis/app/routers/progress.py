from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import ProcessingStatus

router = APIRouter(prefix="/progress", tags=["Progress"])


@router.get("/{filename}")
async def get_progress(filename: str, db: Session = Depends(SessionLocal)):
    status = db.query(ProcessingStatus).filter_by(filename=filename).first()
    if not status:
        return {"error": "File not found"}
    return {"filename": filename, "status": status.status, "progress": status.progress}
