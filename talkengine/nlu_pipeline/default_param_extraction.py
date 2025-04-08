"""Default implementation of parameter extraction interface for TalkEngine.

Provides a basic placeholder implementation.
"""

from typing import Any, Dict

from talkengine.nlu_pipeline.nlu_engine_interfaces import (
    ParameterExtractionInterface,
)
from talkengine.utils.logging import logger


# pylint: disable=unused-argument, too-few-public-methods
class DefaultParameterExtraction(ParameterExtractionInterface):
    """Default parameter extraction - returns empty dict.

    Requires command metadata (for potential future use) to be passed
    during initialization.
    """

    def __init__(self, command_metadata: Dict[str, Any]):
        """Initialize with command metadata.

        Args:
            command_metadata: The dictionary describing available commands,
                              used potentially for parameter type info.
        """
        self._command_metadata = command_metadata
        logger.debug("DefaultParameterExtraction initialized.")

    def identify_parameters(self, user_input: str, intent: str) -> Dict[str, Any]:
        """Default implementation: returns empty parameters.

        Args:
            user_input: The natural language query from the user.
            intent: The classified intent (command key).

        Returns:
            An empty dictionary.
        """
        logger.debug(
            "Default identify_parameters called for intent '%s'. Returning empty dict.",
            intent,
        )
        # Placeholder: A real implementation would parse user_input based on
        # the expected parameters for the intent found in self._command_metadata.
        return {}
