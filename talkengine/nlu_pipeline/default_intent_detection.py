"""Default implementation of intent detection interface for TalkEngine.

Provides a basic keyword/substring matching implementation.
"""

from typing import Any, Dict, List, Optional

from talkengine.nlu_pipeline.nlu_engine_interfaces import IntentDetectionInterface
from ..models import NLUPipelineContext
from talkengine.utils.logging import logger


# pylint: disable=too-few-public-methods
class DefaultIntentDetection(IntentDetectionInterface):
    """Default intent detection using keyword/substring matching.

    Requires command metadata to be passed during initialization.
    """

    def __init__(self, command_metadata: Dict[str, Any]):
        """Initialize with command metadata.

        Args:
            command_metadata: The dictionary describing available commands,
                              used to get command keys and potentially descriptions.
        """
        self._command_metadata = command_metadata
        self._command_keys = list(self._command_metadata.keys())
        logger.debug(
            "DefaultIntentDetection initialized with %d commands.",
            len(self._command_keys),
        )

    def _find_best_match(
        self, commands: list[str], user_input: str
    ) -> tuple[Optional[str], int]:
        """Finds the best command match based on user input keywords/substrings.

        Args:
            commands: List of available command keys.
            user_input: The user's input string.

        Returns:
            A tuple containing the best matched command key (or None) and the match type
            (0=None, 1=Substring, 2=Exact/Word).
        """
        best_match_cmd: Optional[str] = None
        best_match_type: int = 0  # 0=None, 1=Substring, 2=Exact/Word

        user_input_lower = user_input.lower()
        user_input_lower_padded = f" {user_input_lower} "

        # First pass: Prioritize exact/word matches (type 2)
        for cmd in commands:
            # Extract the last part of the command key (function/method name)
            command_name_raw = cmd.split(".")[-1].lower()
            # Create a version with underscores replaced by spaces
            command_name_spaced = command_name_raw.replace("_", " ")

            # Check if the spaced version is present as a whole word/phrase
            if (
                f" {command_name_spaced} " in user_input_lower_padded
                or user_input_lower.startswith(f"{command_name_spaced} ")
                or user_input_lower.endswith(f" {command_name_spaced}")
                or user_input_lower == command_name_spaced
            ):
                # Found the best possible match type (spaced exact/word)
                return cmd, 2

            # Check if the raw version (e.g., "add_todo") is present as whole word
            is_raw_match = (
                f" {command_name_raw} " in user_input_lower_padded
                or user_input_lower.startswith(f"{command_name_raw} ")
                or user_input_lower.endswith(f" {command_name_raw}")
                or user_input_lower == command_name_raw
            )
            if is_raw_match and best_match_type < 2:
                # Found a type 2 raw match, store it but continue searching (prefer spaced version)
                best_match_cmd = cmd
                best_match_type = 2

        # If we found a type 2 raw match in the first pass, return it now
        if best_match_type == 2:
            return best_match_cmd, 2

        # Second pass: Find substring matches (type 1) if no type 2 found
        for cmd in commands:
            command_name_raw = cmd.split(".")[-1].lower()
            command_name_spaced = command_name_raw.replace("_", " ")

            # --- Simple Substring Logic --- #
            # Check if the spaced name is a substring
            if command_name_spaced in user_input_lower:
                return cmd, 1

            # Check if the raw name is a substring
            if command_name_raw in user_input_lower:
                return cmd, 1
            # --- End Simple Substring Logic --- #

            # TODO: Consider using command descriptions from metadata for better matching?

        # No match found
        return None, 0

    def classify_intent(
        self,
        user_input: str,
        context: NLUPipelineContext,
        excluded_intents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Default implementation using keyword/substring matching."""
        logger.debug("Classifying intent for: %s", user_input)
        excluded_set = set(excluded_intents) if excluded_intents else set()

        # Use command keys stored during initialization
        available_commands = [
            cmd for cmd in self._command_keys if cmd not in excluded_set
        ]

        if not available_commands:
            logger.debug("No available commands after excluding: %s", excluded_set)
            intent = "unknown"
            confidence = 0.0
            return {"intent": intent, "confidence": confidence}

        matched_command, match_type = self._find_best_match(
            available_commands, user_input
        )

        if matched_command:
            intent = matched_command
            # Confidence based on match type (simple heuristic)
            confidence = 0.9 if match_type == 2 else 0.7
        else:
            intent = "unknown"
            confidence = 0.1

        result = {"intent": intent, "confidence": confidence}
        logger.debug("Classified intent result: %s", result)
        return result
