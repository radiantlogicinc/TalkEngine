"""Default implementation of parameter extraction interface for TalkEngine.

Provides a basic placeholder implementation.
"""

from typing import Any, Type

from pydantic import BaseModel

from talkengine.nlu_pipeline.nlu_engine_interfaces import (
    ParameterExtractionInterface,
)
from .models import NLUPipelineContext
from talkengine.nlu_pipeline.interaction_models import ValidationRequestInfo
from talkengine.utils.logging import logger


# pylint: disable=unused-argument, too-few-public-methods
class DefaultParameterExtraction(ParameterExtractionInterface):
    """Default parameter extraction - returns empty dict.

    Requires command metadata (for potential future use) to be passed
    during initialization.
    """

    def __init__(self, command_metadata: dict[str, Any]):
        """Initialize with command metadata.

        Args:
            command_metadata: The dictionary describing available commands,
                              (description, parameter_class). Used for parameter checks.
        """
        self._command_metadata = command_metadata
        logger.debug("DefaultParameterExtraction initialized.")

    def identify_parameters(
        self,
        user_input: str,
        intent: str,
        parameter_class: Type[BaseModel],
        context: NLUPipelineContext,
    ) -> tuple[dict[str, Any], list[ValidationRequestInfo]]:
        """Default implementation: placeholder extraction and basic required field check."""
        logger.debug("Default identify_parameters called for intent '%s'.", intent)

        # 1. Placeholder Extraction Logic
        # TODO: Implement actual parameter extraction (e.g., regex, entity recognition)
        # A real implementation should use parameter_class.model_fields to know
        # which fields to look for and their types.
        extracted_parameters: dict[str, Any] = {}

        # 2. Basic Validation Check (Required Parameters using parameter_class)
        validation_requests: list[ValidationRequestInfo] = []
        if required_params := {
            field_name
            for field_name, field_info in parameter_class.model_fields.items()
            if field_info.is_required()
        }:
            logger.debug(
                f"Checking required parameters for intent '{intent}' based on {parameter_class.__name__}: {required_params}"
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
