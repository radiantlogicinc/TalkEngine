"""Default implementation of text generation interface for TalkEngine.

Provides a basic implementation for generating response text.
"""

from typing import Any, Dict, Optional

from talkengine.nlu_pipeline.nlu_engine_interfaces import TextGenerationInterface
from .models import NLUPipelineContext
from talkengine.utils.logging import logger


# pylint: disable=too-few-public-methods
class DefaultTextGeneration(TextGenerationInterface):
    """Default implementation of text generation functionality."""

    def __init__(self):
        """Initialize the DefaultTextGeneration instance."""
        logger.debug("DefaultTextGeneration initialized.")
        # No specific initialization needed for this basic version

    def generate_text(
        self,
        intent: str,
        parameters: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]],
        context: NLUPipelineContext,
    ) -> Optional[str]:
        """Default implementation: generates a basic string representation."""
        logger.debug(
            "Default generate_text called for intent '%s' with parameters: %s and code_result: %s",
            intent,
            parameters,
            artifacts,
        )

        # Always generate a basic text response if intent is known
        if intent == "unknown":
            return "I'm sorry, I didn't understand that."

        param_str = (
            ", ".join([f"{k}='{v}'" for k, v in parameters.items()])
            or "(no parameters)"
        )

        response_text = f"Intent: {intent}, Parameters: {param_str}"

        # Optionally include code execution result
        if artifacts is not None:
            # Simple string representation of the code result dict
            code_result_str = str(artifacts)
            response_text += f", Code Result: {code_result_str}"

        return response_text
