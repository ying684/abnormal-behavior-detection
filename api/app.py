# api/app.py
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from config.settings import settings
from api.routes import video, health
from api.services.recognition import RecognitionService

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting API...")
    RecognitionService.initialize()
    yield
    print("Shutting down API...")

app = FastAPI(
    title="Behavior Recognition API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(video.router, prefix="/api/video", tags=["Video"])

output_dir = settings.data_dir / "outputs"
output_dir.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(output_dir)), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  # Changed from "api.app:app" to "app:app"
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload
    )
