"""
Defines the core TalkEngine class.
"""

# Allow Pydantic models if available, but don't require it
try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = object  # type: ignore # Define BaseModel as object if Pydantic is not installed

from typing import Any, Optional, Dict, List, Tuple, Union

# Assuming simplified types are defined or basic types used
# from .types import CommandMetadataInput, ConversationHistoryInput, NLUOverridesInput, NLURunResult
from .nlu_pipeline.nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    TextGenerationInterface,
)
from .nlu_pipeline.default_intent_detection import DefaultIntentDetection
from .nlu_pipeline.default_param_extraction import DefaultParameterExtraction
from .nlu_pipeline.default_text_generation import DefaultTextGeneration
from .utils.logging import logger


# Type hint for NLU override values (instances implementing interfaces)
NLUImplementation = Union[
    IntentDetectionInterface, ParameterExtractionInterface, TextGenerationInterface
]


class TalkEngine:
    """NLU Pipeline Engine.

    Processes natural language queries based on developer-provided command metadata
    and optional NLU overrides and conversation history.
    """

    _command_metadata: Dict[str, Any]
    _conversation_history: List[Dict[str, Any]]
    _nlu_overrides_config: Dict[str, NLUImplementation]  # Store the input config
    _intent_detector: IntentDetectionInterface
    _param_extractor: ParameterExtractionInterface
    _text_generator: TextGenerationInterface
    _is_trained: bool

    def __init__(
        self,
        command_metadata: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        nlu_overrides: Optional[Dict[str, NLUImplementation]] = None,
    ):
        """Initializes the TalkEngine with configuration.

        Args:
            command_metadata: A dictionary describing the available commands.
                              Example structure:
                              {
                                "command_key1": {"description": "...", "parameters": {"param1": "type"}},
                                "command_key2": {...}
                              }
            conversation_history: An optional list of previous conversation turns.
                                  Example structure: [{ "role": "user", "content": "..." }, ...]
            nlu_overrides: Optional dictionary mapping NLU interface types ('intent_detection',
                           'param_extraction', 'text_generation') to instances implementing
                           the respective interfaces.
                           Example: { "intent_detection": MyIntentDetector(config),
                                     "param_extraction": MyParamExtractor() }
        """
        # Use a helper method for core initialization logic
        self._do_initialize(command_metadata, conversation_history, nlu_overrides)

    def _do_initialize(
        self,
        command_metadata: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        nlu_overrides: Optional[Dict[str, NLUImplementation]] = None,
    ):
        """Core initialization logic, used by __init__ and reset."""
        logger.info("Initializing TalkEngine internals...")
        # Store configuration
        self._command_metadata = command_metadata
        self._conversation_history = (
            conversation_history if conversation_history else []
        )
        # Ensure overrides is a dict
        self._nlu_overrides_config = nlu_overrides if nlu_overrides else {}

        # Placeholder for train() effect
        self._is_trained = False

        # Initialize components based on overrides/defaults
        self._initialize_nlu_components()
        logger.info("TalkEngine Internals Initialized.")

    def _initialize_nlu_components(self):
        """(Internal) Initializes NLU components using defaults and overrides."""
        logger.debug("Initializing NLU components...")

        # Intent Detection
        intent_override = self._nlu_overrides_config.get("intent_detection")
        if isinstance(intent_override, IntentDetectionInterface):
            self._intent_detector = intent_override
            logger.debug("Using provided IntentDetection override.")
        elif intent_override is not None:
            logger.warning(
                "Invalid IntentDetection override provided (type mismatch). Using default."
            )
            self._intent_detector = DefaultIntentDetection(self._command_metadata)
        else:
            logger.debug("Using DefaultIntentDetection.")
            # Pass necessary config to default
            self._intent_detector = DefaultIntentDetection(self._command_metadata)

        # Parameter Extraction
        param_override = self._nlu_overrides_config.get("param_extraction")
        if isinstance(param_override, ParameterExtractionInterface):
            self._param_extractor = param_override
            logger.debug("Using provided ParameterExtraction override.")
        elif param_override is not None:
            logger.warning(
                "Invalid ParameterExtraction override provided (type mismatch). Using default."
            )
            self._param_extractor = DefaultParameterExtraction(self._command_metadata)
        else:
            logger.debug("Using DefaultParameterExtraction.")
            # Pass necessary config to default
            self._param_extractor = DefaultParameterExtraction(self._command_metadata)

        # Text Generation
        text_gen_override = self._nlu_overrides_config.get("text_generation")
        if isinstance(text_gen_override, TextGenerationInterface):
            self._text_generator = text_gen_override
            logger.debug("Using provided TextGeneration override.")
        elif text_gen_override is not None:
            logger.warning(
                "Invalid TextGeneration override provided (type mismatch). Using default."
            )
            self._text_generator = DefaultTextGeneration()
        else:
            logger.debug("Using DefaultTextGeneration.")
            self._text_generator = DefaultTextGeneration()

        logger.debug("NLU components initialized.")

    def train(self) -> None:
        """Configures/trains the internal pipeline components.

        (Currently a placeholder as per requirements).
        """
        logger.info(
            "TalkEngine train() called (Placeholder).",
        )
        # In a real scenario, this might involve:
        # - Validating metadata
        # - Calling train() methods on NLU components if they exist
        # - Setting up classifiers based on descriptions
        # - Analyzing history
        self._is_trained = True
        pass

    def run(self, query: str) -> Tuple[Dict[str, Any], str]:
        """Processes a single natural language query.

        Args:
            query: The user's natural language input.

        Returns:
            A tuple containing:
            1. NLU Result (dict): Contains 'intent', 'parameters', 'raw_response',
               and 'response_text'.
            2. Hint (str): Currently always "new_conversation".
        """
        logger.info(f"TalkEngine run() called with query: '{query}'")

        # --- NLU Pipeline Steps ---
        # 1. Intent Classification
        # TODO: Handle potential exceptions from NLU components
        intent_result = self._intent_detector.classify_intent(query)
        identified_intent = intent_result.get("intent", "unknown_intent")
        confidence = intent_result.get("confidence", 0.0)
        logger.debug(
            f"Intent Classified: {identified_intent} (Confidence: {confidence:.2f})"
        )

        # 2. Parameter Extraction
        # Only extract if intent is known? Or always try? Currently trying always.
        parameters = self._param_extractor.identify_parameters(query, identified_intent)
        logger.debug(f"Parameters Extracted: {parameters}")

        # 3. Text Generation
        raw_response, response_text = self._text_generator.generate_text(
            identified_intent, parameters
        )
        logger.debug(f"Generated Raw Response: {raw_response}")
        logger.debug(f"Generated Response Text: {response_text}")
        # --- End NLU Pipeline Steps ---

        nlu_result: Dict[str, Any] = {
            "intent": identified_intent,
            "parameters": parameters,
            "confidence": confidence,  # Added confidence
            "raw_response": raw_response,
            "response_text": response_text,
        }

        hint = "new_conversation"  # As per requirement

        # Potentially update internal history state here if needed for subsequent runs
        # self._conversation_history.append(...)

        logger.info(f"TalkEngine run() completed. Intent: {identified_intent}")
        return nlu_result, hint

    def reset(
        self,
        command_metadata: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        nlu_overrides: Optional[Dict[str, NLUImplementation]] = None,  # Corrected hint
    ) -> None:
        """Resets and re-initializes the engine with new data."""
        logger.info("Resetting TalkEngine...")
        # Re-run initialization logic using the helper method
        self._do_initialize(command_metadata, conversation_history, nlu_overrides)


# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    print("--- TalkEngine Example --- ")
    # Configure logging for the example
    import logging

    logging.basicConfig(level=logging.DEBUG)

    meta = {
        "calculator.add": {
            "description": "Adds two numbers.",
            "parameters": {"num1": "int", "num2": "int"},
        },
        "weather.get_forecast": {
            "description": "Gets weather.",
            "parameters": {"location": "str", "date": "str"},
        },
        "search.web": {
            "description": "Search the web",
            "parameters": {"search_term": "str"},
        },
    }
    hist = [{"role": "user", "content": "previous query"}]

    # --- Example 1: Using Defaults ---
    print("\n--- Running with Defaults ---")
    engine_default = TalkEngine(command_metadata=meta, conversation_history=hist)
    engine_default.train()

    queries_default = [
        "add 5 and 10",
        "what is the weather in london tomorrow?",
        "tell me a joke",
    ]

    for q in queries_default:
        print(f"\nQuery: {q}")
        result, hint = engine_default.run(q)
        print(f"Hint: {hint}")
        print(f"Result: {result}")

    # --- Example 2: Using Overrides ---
    print("\n--- Running with Overrides ---")

    # Define dummy override functions/classes
    class MyIntentDetector(IntentDetectionInterface):
        def __init__(self, config):
            logger.debug("MyIntentDetector initialized")

        def classify_intent(
            self, user_input: str, excluded_intents=None
        ) -> Dict[str, Any]:
            logger.debug("MyIntentDetector classify running")
            if "search" in user_input:
                return {"intent": "search.web", "confidence": 0.99}
            return {"intent": "override_unknown", "confidence": 0.5}

    class MyParamExtractor(ParameterExtractionInterface):
        def __init__(self, config):
            logger.debug("MyParamExtractor initialized")

        def identify_parameters(self, user_input: str, intent: str) -> Dict[str, Any]:
            logger.debug("MyParamExtractor identify running")
            if intent == "search.web":
                return {"search_term": user_input.split("search for ")[-1]}
            return {"extracted_by": "override"}

    my_overrides: Dict[str, NLUImplementation] = {
        "intent_detection": MyIntentDetector(meta),  # Pass metadata if needed
        "param_extraction": MyParamExtractor(meta),
        # Omitting text_generation override, will use default
    }

    engine_override = TalkEngine(command_metadata=meta, nlu_overrides=my_overrides)
    engine_override.train()
    result_override, hint_override = engine_override.run("search for blue widgets")
    print("\nQuery: search for blue widgets")
    print(f"Hint: {hint_override}")
    print(f"Result: {result_override}")

    # --- Example 3: Resetting ---
    print("\n--- Resetting Engine --- ")
    new_meta = {
        "lights.on": {"description": "Turn lights on", "parameters": {"room": "str"}}
    }
    # Resetting the first engine instance
    engine_default.reset(command_metadata=new_meta)
    engine_default.train()
    result_reset, hint_reset = engine_default.run("turn the kitchen lights on")
    print("\nQuery: turn the kitchen lights on")
    print(f"Hint: {hint_reset}")
    print(f"Result: {result_reset}")
