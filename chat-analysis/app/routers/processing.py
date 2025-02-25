from fastapi import APIRouter, BackgroundTasks, Depends
from app.tasks import process_chat
from sqlalchemy.orm import Session
from app.database import SessionLocal

router = APIRouter(prefix="/process", tags=["Processing"])


@router.post("/{filename}")
async def start_processing(
    filename: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(SessionLocal),
):
    background_tasks.add_task(process_chat, filename, db)
    return {"message": f"Processing started for {filename}"}
