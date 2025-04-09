"""NLU Pipeline Package.

This package contains the core components for Natural Language Understanding (NLU)
in the talkengine system, including interfaces and default implementations.
"""

# Interfaces
from .nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    TextGenerationInterface,
)

# Default Implementations
from .default_intent_detection import DefaultIntentDetection
from .default_param_extraction import DefaultParameterExtraction
from .default_text_generation import DefaultTextGeneration

# Models (If any remain relevant - currently empty/deleted)
# from .models import ...

__all__ = [
    # Interfaces
    "IntentDetectionInterface",
    "ParameterExtractionInterface",
    "TextGenerationInterface",
    # Default Implementations
    "DefaultIntentDetection",
    "DefaultParameterExtraction",
    "DefaultTextGeneration",
    # Models (if added back)
]
