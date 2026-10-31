"""
backend/main.py — Main FastAPI application entry point for Lumarium.
Serves the REST API and the static web frontend.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.health import router as health_router
from backend.api.analyze import router as analyze_router
from backend.api.practice import router as practice_router
from backend.api.guide import router as guide_router

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="Lumarium",
    description="Illuminate every word — Interactive Hybrid NLP Grammar Learning Platform",
    version="1.0.0",
)

# Enable CORS for local development and client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST API endpoints
app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(practice_router)
app.include_router(guide_router)

# Mount frontend directory for seamless local web serving
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8001, reload=True)
