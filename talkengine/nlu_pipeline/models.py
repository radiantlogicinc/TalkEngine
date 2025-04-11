"""Pydantic models and Enums defining the state and context for the NLU pipeline."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict

from .interaction_models import ClarificationData, ValidationData


class NLUPipelineState(str, Enum):
    """Enum defining possible states in the NLU pipeline."""

    INTENT_CLASSIFICATION = "intent_classification"
    INTENT_CLARIFICATION = "intent_clarification"
    PARAMETER_IDENTIFICATION = "parameter_identification"
    PARAMETER_VALIDATION = "parameter_validation"
    CODE_EXECUTION = "code_execution"
    RESPONSE_TEXT_GENERATION = "response_text_generation"


class InteractionState(str, Enum):
    """Enum defining specific interaction modes within NLU states."""

    CLARIFYING_INTENT = "clarifying_intent"
    VALIDATING_PARAMETER = "validating_parameter"
    AWAITING_FEEDBACK = "awaiting_feedback"
    # Add other specific modes as necessary


class NLUPipelineContext(BaseModel):
    """Model for storing NLU pipeline state and context."""

    # Config/Static info (passed during init)
    command_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)

    # Pipeline state/tracking
    current_state: NLUPipelineState = NLUPipelineState.INTENT_CLASSIFICATION
    excluded_intents: list[str] = Field(default_factory=list)
    current_intent: Optional[str] = None
    current_parameters: dict[str, Any] = Field(default_factory=dict)
    parameter_validation_errors: list[str] = Field(default_factory=list)
    classroom_mode: bool = False
    confidence_score: float = 0.0
    # Add fields to store info needed for response refinement on feedback
    last_user_message_for_response: Optional[str] = None
    last_artifacts_for_response: Optional[dict[str, Any]] = None
    artifacts: Optional[BaseModel] = None

    # New fields for modal interaction
    interaction_mode: Optional[InteractionState] = None
    interaction_data: Optional[ClarificationData | ValidationData] = None

    # Use modern ConfigDict instead of inner class Config
    model_config = ConfigDict(arbitrary_types_allowed=True)
