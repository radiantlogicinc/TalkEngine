from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from .types import InteractionLogEntry  # Import the type alias


class ConversationDetail(BaseModel):
    """Details the interaction flow and final response for one NLU processing attempt."""

    interactions: List[InteractionLogEntry] = Field(default_factory=list)
    response_text: Optional[str] = None


class NLUResult(BaseModel):
    """Structured result returned by TalkEngine.run for a single processing attempt."""

    command: Optional[str] = (
        None  # Changed to Optional, might be None during early interaction return
    )
    parameters: Optional[Dict[str, Any]] = Field(default_factory=lambda: {})
    artifacts: Optional[Dict[str, Any]] = None
    conversation_detail: ConversationDetail
