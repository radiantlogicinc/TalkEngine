"""Default implementation of text generation interface for TalkEngine.

Provides a basic implementation for generating responses.
"""

from typing import Any, Dict, Tuple

from talkengine.nlu_pipeline.nlu_engine_interfaces import TextGenerationInterface
from talkengine.utils.logging import logger


# pylint: disable=too-few-public-methods
class DefaultTextGeneration(TextGenerationInterface):
    """Default implementation of text generation functionality."""

    def __init__(self):
        """Initialize the DefaultTextGeneration instance."""
        logger.debug("DefaultTextGeneration initialized.")
        # No specific initialization needed for this basic version

    def generate_text(self, intent: str, parameters: Dict[str, Any]) -> Tuple[Any, str]:
        """Default implementation: generates a basic dictionary and string representation."""
        logger.debug(
            "Default generate_text called for intent '%s' with parameters: %s",
            intent,
            parameters,
        )

        # Raw response is just the intent and parameters
        raw_response: Dict[str, Any] = {"intent": intent, "parameters": parameters}

        # Response text is a simple string representation
        param_str = ", ".join([f"{k}='{v}'" for k, v in parameters.items()])
        if not param_str:
            param_str = "(no parameters)"
        response_text = f"Intent: {intent}, Parameters: {param_str}"

        return raw_response, response_text
