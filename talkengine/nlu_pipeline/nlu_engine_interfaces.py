"""Simplified Natural Language Understanding (NLU) engine interfaces for TalkEngine.

Defines abstract interfaces for core NLU components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

# Import necessary types - Corrected path
from ..models import NLUPipelineContext
from .interaction_models import ValidationRequestInfo


# pylint: disable=too-few-public-methods
class IntentDetectionInterface(ABC):
    """Interface for classifying user intent.

    Implementations should receive necessary context (like command descriptions)
    during initialization.
    """

    @abstractmethod
    def classify_intent(
        self,
        user_input: str,
        context: NLUPipelineContext,
        excluded_intents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify user intent based on input.

        Args:
            user_input: The natural language query from the user.
            context: The current NLU pipeline context.
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
    def identify_parameters(
        self,
        user_input: str,
        intent: str,
        context: NLUPipelineContext,
    ) -> Tuple[Dict[str, Any], List[ValidationRequestInfo]]:
        """Extract parameters from user input for the given classified intent.

        Args:
            user_input: The natural language query from the user.
            intent: The classified intent (command key).
            context: The current NLU pipeline context.

        Returns:
            A tuple containing:
            1. Dictionary of extracted parameter names and their values.
               Example: {"param1": "value1", "param2": 123}
            2. List of ValidationRequestInfo objects detailing parameters
               that require user validation (e.g., missing required fields).
               Example: [ValidationRequestInfo(parameter_name='loc', reason='missing_required')]
        """
        # Default implementation should return ({}, [])
        raise NotImplementedError


# pylint: disable=too-few-public-methods
class TextGenerationInterface(ABC):
    """Interface for generating user-facing response text.

    Implementations might use simple formatting or more complex generation logic.
    They receive the intent, parameters, and potentially the result of any
    executed command code.
    """

    @abstractmethod
    def generate_text(
        self,
        intent: str,
        parameters: Dict[str, Any],
        code_execution_result: Optional[Dict[str, Any]],
        context: NLUPipelineContext,
    ) -> Optional[str]:
        """Generate a user-facing text response.

        Args:
            intent: The classified intent (command key).
            parameters: The extracted parameters.
            code_execution_result: Result from executing code associated with the
                                     command, if any.
            context: The current NLU pipeline context.

        Returns:
            A user-friendly string representation, or None if no text is generated.
        """
        # Default implementation could return a simple string or None
        raise NotImplementedError
