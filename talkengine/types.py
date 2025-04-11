"""
Simplified type definitions for TalkEngine.
"""

from typing import Any, Callable, TypeAlias, Optional, Union, Type

from pydantic import BaseModel

# Import NLU Interfaces for type hints
from .nlu_pipeline.nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    TextGenerationInterface,
)

# Type alias for NLU implementation instances
NLUImplementation = Union[
    IntentDetectionInterface, ParameterExtractionInterface, TextGenerationInterface
]

# Basic value types that can be passed as parameters or context
ParamValue = str | bool | int | float
# Allow more complex types, especially for parameters returned by NLU
ExtendedParamValue = (
    str | bool | int | float | BaseModel | dict[str, Any] | list[Any] | None
)

# --- NEW Type Aliases for Configuration --- #

# Type alias for the structure defining executable code override
ExecutableCodeOverride: TypeAlias = dict[str, Callable | Type[BaseModel]]

# Type alias for the override structure specific to a single command
CommandOverride: TypeAlias = dict[str, ExecutableCodeOverride]

# Type alias for the complete developer-provided NLU overrides configuration dictionary
NLUOverridesConfig: TypeAlias = dict[str, NLUImplementation | CommandOverride]

# Type alias for the complete developer-provided command metadata configuration dictionary
CommandMetadataConfig: TypeAlias = dict[str, dict[str, str | Type[BaseModel]]]

# --- Original Type Aliases --- #

# Type alias for the developer-provided conversation history list
ConversationHistoryInput: TypeAlias = list[dict[str, Any]]

# Type alias for interaction log entries used in ConversationDetail
# Structure: (interaction_stage_type: str, prompt_shown: str, user_response: Optional[str])
InteractionLogEntry: TypeAlias = tuple[str, str, Optional[str]]

# Deprecated/Removed aliases:
# CommandMetadataInput: TypeAlias = dict[str, Any]
# NLUOverridesInput: TypeAlias = dict[str, dict[str, Callable[..., Any]]]
# NLURunResult: TypeAlias = dict[str, Any]

__all__ = [
    "ParamValue",
    "ExtendedParamValue",
    "CommandMetadataConfig",
    "NLUOverridesConfig",
    "ExecutableCodeOverride",
    "CommandOverride",
    "NLUImplementation",
    "ConversationHistoryInput",
    "InteractionLogEntry",
]
