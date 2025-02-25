from fastapi import FastAPI
from app.routers import upload, processing, progress, results
import uvicorn

app = FastAPI(
    title="Chat Analytics API", description="Analyzes chat data using NLP models."
)

# Include routers
app.include_router(upload.router)
app.include_router(processing.router)
app.include_router(progress.router)
app.include_router(results.router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
