"""Core data models for TalkEngine requests and results."""

from typing import Optional, Any, NamedTuple
from pydantic import BaseModel, Field, ConfigDict

# from .nlu_pipeline.models import InteractionState # Causes circular import


# Define InteractionLogEntry as a NamedTuple for structure and keyword args
class InteractionLogEntry(NamedTuple):
    """Represents a single interaction entry in the conversation detail."""

    stage: str  # Corresponds to InteractionState.value
    prompt: Optional[str]
    response: Optional[str]


# Old tuple definition - Replace with NamedTuple above
# InteractionLogEntry = tuple[str, Optional[str], Optional[str]] # stage, prompt, response


class ConversationDetail(BaseModel):
    """Details the interaction flow and final response for one NLU processing attempt."""

    interactions: list[InteractionLogEntry] = Field(default_factory=list)
    response_text: Optional[str] = None


class NLUResult(BaseModel):
    """Structured result returned by TalkEngine.run for a single processing attempt."""

    command: Optional[str] = (
        None  # Changed to Optional, might be None during early interaction return
    )
    parameters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Extracted parameter values corresponding to fields in the command's parameter_class.",
    )
    artifacts: Optional[BaseModel] = Field(
        default=None,
        description="Pydantic model instance returned by executable_code, if any.",
    )
    conversation_detail: ConversationDetail = Field(default_factory=ConversationDetail)

    model_config = ConfigDict(arbitrary_types_allowed=True)
