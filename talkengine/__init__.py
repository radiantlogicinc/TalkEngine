"""
TalkEngine NLU Pipeline Library.
"""

import os
from typing import Optional

# Import the core engine class
from .engine import TalkEngine
from .types import ExtendedParamValue

# Environment variable cache (can potentially be removed if not widely used by NLU overrides)
_env_vars: dict[str, str] = {}


# Note: @command decorator is no longer functional for discovery.
# It might be kept purely for informational purposes or removed entirely.
def command(func):
    """(Informational) Decorator to conceptually mark a command function."""
    return func


def get_env_var(
    var_name: str,
    var_type: type = str,
    default: Optional[ExtendedParamValue] = None,
) -> ExtendedParamValue:
    """Get an environment variable, checking a local cache first.

    Args:
        var_name: Name of the environment variable to get
        var_type: Type to convert the value to (str, int, float, or bool)
        default: Optional default value if variable is not found

    Returns:
        The environment variable value converted to the specified type

    Raises:
        ValueError: If the variable doesn't exist and no default is provided,
                  or if the value cannot be converted to the specified type
    """
    value = _env_vars.get(var_name)
    if value is None:
        if default is not None:
            return default
        value = os.getenv(var_name)

    if value is None:
        raise ValueError(
            f"Environment variable '{var_name}' does not exist and no default value is provided."
        )

    try:
        if var_type is int:
            return int(value)
        if var_type is float:
            return float(value)
        if var_type is bool:
            if value.lower() in ("true", "1"):
                return True
            if value.lower() in ("false", "0"):
                return False
            raise ValueError(f"Cannot convert '{value}' to {var_type.__name__}.")
        return str(value)  # Default case for str
    except ValueError as e:
        raise ValueError(f"Cannot convert '{value}' to {var_type.__name__}.") from e


__all__ = ["TalkEngine", "get_env_var"]
