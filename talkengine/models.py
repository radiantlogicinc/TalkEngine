from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, ConfigDict
import enum

# Assuming NLUPipelineContext might already exist, we'll add to it.
# If the file is new or doesn't have it, this structure should be fine.


class InteractionState(enum.Enum):
    """Represents the current state of interaction within the NLU pipeline."""

    IDLE = "idle"  # Default state, not in an interaction
    CLARIFYING_INTENT = "clarifying_intent"
    VALIDATING_PARAMETER = "validating_parameter"
    # Add other states as needed, e.g., AWAITING_FEEDBACK


# Define a type alias for interaction log entries for clarity
# Structure: (interaction_stage_type: str, prompt_shown: str, user_response: Optional[str])
InteractionLogEntry = Tuple[str, str, Optional[str]]


class NLUPipelineContext(BaseModel):
    """Holds the state and context throughout a single NLU pipeline run."""

    current_intent: Optional[str] = None
    current_parameters: Dict[str, Any] = Field(default_factory=dict)
    current_confidence: Optional[float] = None
    interaction_mode: InteractionState = InteractionState.IDLE
    interaction_data: Optional[Any] = (
        None  # Holds data needed for the current interaction (e.g., ClarificationData)
    )
    recorded_interactions: List[InteractionLogEntry] = Field(default_factory=list)
    last_prompt_shown: Optional[str] = (
        None  # Added to track the last prompt for logging
    )
    # Add other context fields if necessary (e.g., session_id, user_id)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ConversationDetail(BaseModel):
    """Details the interaction flow and final response for one NLU processing attempt."""

    interactions: List[InteractionLogEntry] = Field(default_factory=list)
    response_text: Optional[str] = None


class NLUResult(BaseModel):
    """Structured result returned by TalkEngine.run for a single processing attempt."""

    command: Optional[str] = (
        None  # Changed to Optional, might be None during early interaction return
    )
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    confidence: Optional[float] = None
    code_execution_result: Optional[Dict[str, Any]] = None
    conversation_detail: ConversationDetail
