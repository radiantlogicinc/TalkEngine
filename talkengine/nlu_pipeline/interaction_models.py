"""Pydantic models for interaction data used within the NLU pipeline."""

from typing import Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field


# pylint: disable=too-few-public-methods
class BaseInteractionData(BaseModel):
    """Base model for data passed to interaction handlers."""

    # user_input: str # Removed - No longer needed here


class ClarificationData(BaseModel):
    """Data needed for intent clarification."""

    options: list[str]  # list of intent names or descriptions to choose from
    # original_query: str # Removed - No longer needed here
    prompt: str = "Please choose one of the following options:"  # Default prompt


class FeedbackData(BaseInteractionData):
    """Data needed for response feedback."""

    response_text: str
    artifacts: Optional[dict[str, Any]] = None
    prompt: str = "Was this response helpful? (yes/no/details)"  # Default prompt


# Structure to signal validation needs from Parameter Extractor
@dataclass
class ValidationRequestInfo:
    """Details about a parameter requiring validation."""

    parameter_name: str
    reason: str = Field(..., description="Reason code for why validation is needed.")
    current_value: Optional[Any] = (
        None  # Optional current value if ambiguity needs resolving
    )


class ValidationData(BaseModel):
    """Holds data needed during parameter validation interaction."""

    requests: list[ValidationRequestInfo] = Field(
        ..., description="List of parameters needing validation."
    )
    prompt: Optional[str] = Field(
        None, description="The prompt presented to the user, set by the handler."
    )
    # Removed inheritance from ValidationRequestInfo
    # parameter_name: str inherited field is incorrect here
    # reason: str inherited field is incorrect here


# Add other interaction data models as needed
