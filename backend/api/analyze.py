"""
backend/api/analyze.py — Primary POS analysis and linguistic enrichment endpoint.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.nlp_service import nlp_service

router = APIRouter(prefix="/api", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    text: str
    threshold: Optional[float] = None
    fallback: Optional[str] = None
    rule_overridable: Optional[bool] = None
    use_transformer: bool = True


@router.post("/analyze")
def analyze_text(payload: AnalyzeRequest):
    """
    Executes the multi-tier hybrid POS tagger, machine translation, morphology,
    language identification, NER, and WordNet validation on the input text.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    result = nlp_service.analyze_sentence(
        text=payload.text,
        threshold=payload.threshold,
        fallback=payload.fallback,
        rule_overridable=payload.rule_overridable,
        use_transformer=payload.use_transformer,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result
