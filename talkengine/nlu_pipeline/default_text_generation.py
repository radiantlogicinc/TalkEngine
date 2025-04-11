"""Default implementation of text generation interface for TalkEngine.

Provides a basic implementation for generating response text.
"""

from typing import Any, Optional

from pydantic import BaseModel

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
        *,
        command: Optional[str] = None,
        parameters: dict[str, Any],
        artifacts: Optional[BaseModel] = None,
        context: NLUPipelineContext,
        **kwargs: Any,
    ) -> str:
        """Generates a simple text representation of the NLU result."""
        # logger.debug(
        #     f"DefaultTextGeneration generate_text called. Command: {command}, "
        #     f"Params: {parameters}, Artifacts: {artifacts is not None}"
        # )
        param_str = (
            f"Parameters: {parameters}" if parameters else "Parameters: (no parameters)"
        )
        # Include command if provided
        command_str = f"Intent: {command}" if command else "Intent: (not determined)"
        artifact_str = (
            f"Artifacts: {type(artifacts).__name__}" if artifacts else "Artifacts: None"
        )
        # return f"Intent: {context.current_intent or '(not determined)'}, {param_str}, Artifacts: {artifact_str}"
        return f"{command_str}, {param_str}, {artifact_str}"


# --- How to Use --- #
