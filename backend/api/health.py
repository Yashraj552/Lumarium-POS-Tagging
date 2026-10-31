"""
backend/api/health.py — Health check and model status diagnostic endpoint.
"""
from fastapi import APIRouter
from backend.services.nlp_service import nlp_service

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def get_health():
    """Returns application status and diagnostic health of all underlying NLP tiers."""
    return nlp_service.get_health_status()
