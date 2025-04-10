"""Pydantic models for interaction data used within the NLU pipeline."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from pydantic import BaseModel


# pylint: disable=too-few-public-methods
class BaseInteractionData(BaseModel):
    """Base model for data passed to interaction handlers."""

    # user_input: str # Removed - No longer needed here


class ClarificationData(BaseInteractionData):
    """Data needed for intent clarification."""

    options: List[str]  # List of intent names or descriptions to choose from
    # original_query: str # Removed - No longer needed here
    prompt: str = "Please choose one of the following options:"  # Default prompt


class ValidationData(BaseInteractionData):
    """Data needed for parameter validation."""

    parameter_name: str
    # error_message: str # Removed - Engine generates prompt contextually now
    reason: str  # Added reason from ValidationRequestInfo
    current_value: Optional[Any] = None
    prompt: str = (
        "Please provide a valid value for {parameter_name}:"  # Default prompt template
    )


class FeedbackData(BaseInteractionData):
    """Data needed for response feedback."""

    response_text: str
    artifacts: Optional[Dict[str, Any]] = None
    prompt: str = "Was this response helpful? (yes/no/details)"  # Default prompt


# Structure to signal validation needs from Parameter Extractor
@dataclass
class ValidationRequestInfo:
    """Details about a parameter requiring validation."""

    parameter_name: str
    reason: str  # E.g., "missing_required", "invalid_format", "ambiguous_value"
    current_value: Optional[Any] = (
        None  # Optional current value if ambiguity needs resolving
    )


# Add other interaction data models as needed
