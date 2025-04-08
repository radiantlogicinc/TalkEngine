"""Default implementation of parameter extraction interface for TalkEngine.

Provides a basic placeholder implementation.
"""

from typing import Any, Dict, List
from typing import Tuple

from talkengine.nlu_pipeline.nlu_engine_interfaces import (
    ParameterExtractionInterface,
)
from talkengine.nlu_pipeline.models import NLUPipelineContext
from talkengine.nlu_pipeline.interaction_models import ValidationRequestInfo
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

    def identify_parameters(
        self,
        user_input: str,
        intent: str,
        context: NLUPipelineContext,
    ) -> Tuple[Dict[str, Any], List[ValidationRequestInfo]]:
        """Default implementation: placeholder extraction and basic validation check."""
        logger.debug("Default identify_parameters called for intent '%s'.", intent)

        # 1. Placeholder Extraction Logic
        # TODO: Implement actual parameter extraction (e.g., regex, entity recognition)
        # For now, just return an empty dictionary.
        extracted_parameters: Dict[str, Any] = {}

        # 2. Basic Validation Check (Required Parameters)
        validation_requests: List[ValidationRequestInfo] = []
        intent_meta = self._command_metadata.get(intent, {})
        required_params = intent_meta.get("required_parameters", [])

        if required_params:
            logger.debug(
                f"Checking required parameters for intent '{intent}': {required_params}"
            )
            for param_name in required_params:
                if param_name not in extracted_parameters:
                    logger.warning(
                        f"Required parameter '{param_name}' missing for intent '{intent}'."
                    )
                    validation_requests.append(
                        ValidationRequestInfo(
                            parameter_name=param_name, reason="missing_required"
                        )
                    )

        logger.debug(f"Returning parameters: {extracted_parameters}")
        logger.debug(f"Returning validation requests: {validation_requests}")

        return extracted_parameters, validation_requests
