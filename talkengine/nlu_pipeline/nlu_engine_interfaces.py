"""Simplified Natural Language Understanding (NLU) engine interfaces for TalkEngine.

Defines abstract interfaces for core NLU components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


# pylint: disable=too-few-public-methods
class IntentDetectionInterface(ABC):
    """Interface for classifying user intent.

    Implementations should receive necessary context (like command descriptions)
    during initialization.
    """

    @abstractmethod
    def classify_intent(
        self, user_input: str, excluded_intents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Classify user intent based on input.

        Args:
            user_input: The natural language query from the user.
            excluded_intents: Optional list of intents to exclude from consideration.

        Returns:
            A dictionary containing at least 'intent' (str) and 'confidence' (float).
            Example: {"intent": "command.key", "confidence": 0.85}
        """
        # Default implementation should return {"intent": "unknown", "confidence": 0.0}
        raise NotImplementedError


# pylint: disable=too-few-public-methods
class ParameterExtractionInterface(ABC):
    """Interface for extracting parameters for a given intent.

    Implementations should receive necessary context (like command parameter specs)
    during initialization.
    """

    @abstractmethod
    def identify_parameters(self, user_input: str, intent: str) -> Dict[str, Any]:
        """Extract parameters from user input for the given classified intent.

        Args:
            user_input: The natural language query from the user.
            intent: The classified intent (command key).

        Returns:
            A dictionary of extracted parameter names and their values.
            Example: {"param1": "value1", "param2": 123}
        """
        # Default implementation should return an empty dict
        raise NotImplementedError


# pylint: disable=too-few-public-methods
class TextGenerationInterface(ABC):
    """Interface for generating response text and raw response data.

    Implementations might use simple formatting or more complex generation logic.
    """

    @abstractmethod
    def generate_text(self, intent: str, parameters: Dict[str, Any]) -> Tuple[Any, str]:
        """Generate raw response data and a user-facing text response.

        Args:
            intent: The classified intent (command key).
            parameters: The extracted parameters.

        Returns:
            A tuple containing:
            1. Raw Response (Any): Structured data representing the NLU outcome
               (e.g., a dictionary with intent and parameters).
            2. Response Text (str): A user-friendly string representation.
        """
        # Default implementation could return ({"intent": intent, "parameters": parameters}, str(parameters))
        raise NotImplementedError
