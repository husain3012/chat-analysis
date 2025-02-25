from sqlalchemy import Column, Integer, String, DateTime, Float
from app.database import Base


class ChatFile(Base):
    __tablename__ = "chat_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False)
    uploaded_at = Column(DateTime, nullable=False)


class ProcessingStatus(Base):
    __tablename__ = "processing_status"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    updated_at = Column(DateTime, nullable=False)
