"""
Simplified type definitions for TalkEngine.
"""

from typing import Any, Callable, TypeAlias

from pydantic import BaseModel

# Basic value types that can be passed as parameters or context
ParamValue = str | bool | int | float
# Allow more complex types, especially for parameters returned by NLU
ExtendedParamValue = (
    str | bool | int | float | BaseModel | dict[str, Any] | list[Any] | None
)

# Type alias for the developer-provided command metadata dictionary
CommandMetadataInput: TypeAlias = dict[str, Any]

# Type alias for the developer-provided conversation history list
ConversationHistoryInput: TypeAlias = list[dict[str, Any]]

# Type alias for the developer-provided NLU override functions
# Example: { "command_key": { "intent_detection": my_func, "param_extraction": other_func } }
NLUOverridesInput: TypeAlias = dict[str, dict[str, Callable[..., Any]]]

# Type alias for the result dictionary returned by TalkEngine.run()
NLURunResult: TypeAlias = dict[str, Any]

__all__ = [
    "ParamValue",
    "ExtendedParamValue",
    "CommandMetadataInput",
    "ConversationHistoryInput",
    "NLUOverridesInput",
    "NLURunResult",
]
