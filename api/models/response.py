# api/models/response.py
from pydantic import BaseModel
from typing import Dict, Any, List

class VideoProcessResponse(BaseModel):
    success: bool
    upload_id: str
    output_url: str
    info: Dict[str, Any]

class BatchProcessResponse(BaseModel):
    success: bool
    total: int
    processed: int
    results: List[Dict[str, Any]]
