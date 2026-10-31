"""
backend/api/practice.py — Dynamic interactive quiz generation endpoint based on NLP outputs.
"""
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.nlp_service import nlp_service

router = APIRouter(prefix="/api", tags=["practice"])


class PracticeRequest(BaseModel):
    text: Optional[str] = None


@router.post("/practice")
def generate_practice(payload: PracticeRequest):
    """Generates quiz questions dynamically using real NLP pipeline output."""
    sentence = payload.text if (payload.text and payload.text.strip()) else "Maine kal school mein football khela."
    analysis = nlp_service.analyze_sentence(sentence)
    questions = nlp_service.generate_quiz(analysis)
    return {
        "sentence": sentence,
        "questions": questions,
    }


@router.get("/practice")
def get_default_practice():
    """Generates default practice questions."""
    sentence = "Maine kal school mein football khela."
    analysis = nlp_service.analyze_sentence(sentence)
    questions = nlp_service.generate_quiz(analysis)
    return {
        "sentence": sentence,
        "questions": questions,
    }
