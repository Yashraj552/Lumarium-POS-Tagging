"""
backend/api/guide.py — Universal Dependencies grammar compendium reference endpoint.
"""
from fastapi import APIRouter
from backend.services.nlp_service import nlp_service

router = APIRouter(prefix="/api", tags=["guide"])


@router.get("/guide")
def get_grammar_guide():
    """Returns the full Universal Dependencies educational grammar guide."""
    return nlp_service.get_grammar_guide()
