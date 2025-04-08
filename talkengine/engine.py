"""
Defines the core TalkEngine class.
"""

from typing import Any, Optional, Dict, List, Tuple, Union

# Assuming simplified types are defined or basic types used
# from .types import CommandMetadataInput, ConversationHistoryInput, NLUOverridesInput, NLURunResult
from .nlu_pipeline.nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    ResponseGenerationInterface,
)

# Import NLU models and interaction components
from .nlu_pipeline.models import NLUPipelineContext, InteractionState
from .nlu_pipeline.interaction_handlers import (
    InteractionHandler,
    ClarificationHandler,
    ValidationHandler,
    FeedbackHandler,
    InteractionResult,
)
from .nlu_pipeline.default_intent_detection import DefaultIntentDetection
from .nlu_pipeline.default_param_extraction import DefaultParameterExtraction
from .nlu_pipeline.default_response_generation import DefaultResponseGeneration
from .utils.logging import logger


# Type hint for NLU override values (instances implementing interfaces)
NLUImplementation = Union[
    IntentDetectionInterface, ParameterExtractionInterface, ResponseGenerationInterface
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
    _text_generator: ResponseGenerationInterface
    _is_trained: bool
    # Add pipeline context and interaction handlers
    _pipeline_context: NLUPipelineContext
    _interaction_handlers: Dict[InteractionState, InteractionHandler]

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
        """Initializes or re-initializes the engine components."""
        # Basic validation
        if not isinstance(command_metadata, dict):
            raise ValueError("command_metadata must be a dictionary.")
        if conversation_history is not None and not isinstance(
            conversation_history, list
        ):
            raise ValueError("conversation_history must be a list if provided.")

        self._command_metadata = command_metadata
        self._conversation_history = conversation_history or []
        self._nlu_overrides_config = nlu_overrides or {}

        # Initialize pipeline context
        self._pipeline_context = NLUPipelineContext(
            command_metadata=self._command_metadata,
            conversation_history=self._conversation_history,
            # Other fields will be updated during the pipeline run
        )

        # Initialize NLU components (intent detection, parameter extraction, text generation)
        # using defaults or overrides
        self._initialize_nlu_components()

        # Initialize Interaction Handlers
        self._initialize_interaction_handlers()

        # Set trained status - assume not trained until train() is explicitly called
        self._is_trained = False

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
        if isinstance(text_gen_override, ResponseGenerationInterface):
            self._text_generator = text_gen_override
            logger.debug("Using provided TextGeneration override.")
        elif text_gen_override is not None:
            logger.warning(
                "Invalid TextGeneration override provided (type mismatch). Using default."
            )
            self._text_generator = DefaultResponseGeneration()
        else:
            logger.debug("Using DefaultResponseGeneration.")
            self._text_generator = DefaultResponseGeneration()

        logger.debug("NLU components initialized.")

    def _initialize_interaction_handlers(self):
        """(Internal) Initializes interaction handler instances."""
        logger.debug("Initializing Interaction Handlers...")
        self._interaction_handlers = {
            InteractionState.CLARIFYING_INTENT: ClarificationHandler(),
            InteractionState.VALIDATING_PARAMETER: ValidationHandler(),
            InteractionState.AWAITING_FEEDBACK: FeedbackHandler(),
        }
        logger.debug("Interaction Handlers Initialized.")

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
        """Processes a single natural language query, potentially involving interaction.

        This method implements the core Finite State Machine (FSM) logic.
        If the engine is in an interaction mode (e.g., clarifying intent, validating
        parameters), it delegates the user's query to the appropriate interaction
        handler. The handler processes the input and returns an InteractionResult,
        which determines the next state (staying in the mode, exiting, updating context).

        If not in an interaction mode, it executes the standard NLU pipeline via
        `_run_nlu_pipeline()`. This internal method handles intent classification,
        parameter extraction, and response generation, and includes decision points
        to potentially *enter* an interaction mode based on NLU results (e.g., low
        confidence intent, missing required parameters).

        Args:
            query: The user's natural language input.

        Returns:
            A tuple containing:
            1. Result (dict): Contains the final NLU result OR the interaction prompt data.
               - For final NLU result: Includes keys like 'intent', 'parameters', 'confidence',
                 'raw_response', 'response_text'.
               - For interaction prompt: Includes keys like 'interaction_prompt' (str),
                 'interaction_mode' (InteractionState).
               - For errors: Includes an 'error' key with a description.
            2. Hint (str): Describes the outcome or current state (e.g., 'new_conversation',
               'awaiting_clarification', 'awaiting_validation', 'interaction_clarifying_intent_ended').
        """
        logger.info(f"TalkEngine run() called with query: '{query}'")
        logger.debug(f"Initial context: {self._pipeline_context}")

        # --- FSM Logic ---

        # Check if we are currently in an interaction mode
        if self._pipeline_context.interaction_mode:
            handler = self._interaction_handlers.get(
                self._pipeline_context.interaction_mode
            )
            if not handler:
                logger.error(
                    f"No handler found for interaction mode: {self._pipeline_context.interaction_mode}"
                    + ". Exiting interaction."
                )
                # Clear interaction state to prevent getting stuck
                self._pipeline_context.interaction_mode = None
                self._pipeline_context.interaction_data = None
                return {"error": "Internal handler error"}, "error"

            logger.debug(
                f"In interaction mode: {self._pipeline_context.interaction_mode}. Handling input..."
            )
            interaction_result: InteractionResult = handler.handle_input(
                query, self._pipeline_context
            )
            logger.debug(f"Interaction Result: {interaction_result}")

            # Process InteractionResult
            response_payload: Dict[str, Any]
            hint: str

            # Update context FIRST
            if interaction_result.update_context:
                logger.debug(f"Updating context: {interaction_result.update_context}")
                # Need to handle potential Pydantic model update correctly
                # This is a basic update; might need model_copy/update for Pydantic V2
                for key, value in interaction_result.update_context.items():
                    setattr(self._pipeline_context, key, value)
                # self._pipeline_context = self._pipeline_context.model_copy(update=interaction_result.update_context)

            if interaction_result.exit_mode:
                logger.info(
                    f"Exiting interaction mode: {self._pipeline_context.interaction_mode}"
                )
                current_mode = (
                    self._pipeline_context.interaction_mode
                )  # Store before clearing
                # Clear interaction state
                self._pipeline_context.interaction_mode = None
                self._pipeline_context.interaction_data = None

                if interaction_result.proceed_immediately:
                    logger.info("Proceeding immediately after interaction.")
                    # --- Updated Logic: Call subsequent pipeline steps ---
                    interaction_payload: Optional[Tuple[Dict[str, Any], str]] = None
                    final_result: Dict[str, Any]
                    final_hint: str

                    if current_mode == InteractionState.CLARIFYING_INTENT:
                        logger.debug(
                            "Proceeding after clarification: Running param extraction/validation."
                        )
                        # Run param extraction + validation check
                        interaction_payload = (
                            self._extract_parameters_and_handle_validation(query)
                        )
                        if interaction_payload:
                            # Entered validation mode
                            return interaction_payload[0], interaction_payload[1]
                        else:
                            # Param extraction OK, proceed to response generation
                            logger.debug(
                                "Proceeding after clarification/param extraction: Running response generation."
                            )
                            final_result, final_hint = self._generate_final_response()
                            return final_result, final_hint

                    elif current_mode == InteractionState.VALIDATING_PARAMETER:
                        logger.debug(
                            "Proceeding after validation: Running response generation."
                        )
                        # Validation successful, context updated, just generate response
                        final_result, final_hint = self._generate_final_response()
                        return final_result, final_hint

                    # Add handling for other modes if they can proceed_immediately
                    else:
                        logger.warning(
                            f"Unhandled proceed_immediately case for mode: {current_mode}"
                        )
                        # Fallback: Re-run full pipeline (original problematic behavior)
                        # Consider returning an error or specific state instead.
                        nlu_result, hint = self._run_nlu_pipeline(query)  # Fallback
                        return nlu_result, hint
                    # --- End Updated Logic ---

                else:
                    # Interaction ended, just return the handler's last response
                    response_payload = {"response_text": interaction_result.response}
                    hint = f"interaction_{current_mode.name.lower()}_ended"
                    logger.debug(
                        f"Interaction ended. Response: {response_payload}, Hint: {hint}"
                    )
                    return response_payload, hint
            else:
                # Still in interaction mode, return the handler's response (re-prompt)
                response_payload = {
                    "interaction_prompt": interaction_result.response,
                    "interaction_mode": self._pipeline_context.interaction_mode,
                }
                hint = (
                    f"awaiting_{self._pipeline_context.interaction_mode.name.lower()}"
                )
                logger.debug(
                    f"Staying in interaction. Response: {response_payload}, Hint: {hint}"
                )
                return response_payload, hint

        # --- If NOT in interaction mode, run the standard NLU Pipeline --- #
        else:
            logger.debug("Not in interaction mode. Running standard NLU pipeline.")
            # --- Updated Logic: Call pipeline steps sequentially ---
            pipeline_interaction_payload: Optional[Tuple[Dict[str, Any], str]] = None

            # 1. Classify Intent and Handle Clarification
            pipeline_interaction_payload = (
                self._classify_intent_and_handle_clarification(query)
            )
            if pipeline_interaction_payload:
                return (
                    pipeline_interaction_payload[0],
                    pipeline_interaction_payload[1],
                )  # Return interaction prompt

            # 2. Extract Parameters and Handle Validation
            pipeline_interaction_payload = (
                self._extract_parameters_and_handle_validation(query)
            )
            if pipeline_interaction_payload:
                return (
                    pipeline_interaction_payload[0],
                    pipeline_interaction_payload[1],
                )  # Return interaction prompt

            # 3. Generate Final Response
            final_result, final_hint = self._generate_final_response()
            return final_result, final_hint

    def _run_nlu_pipeline(self, query: str) -> Tuple[Dict[str, Any], str]:
        """(Internal) Executes the core NLU pipeline steps by calling helper methods.

        DEPRECATED in favor of calling steps directly, but kept for potential fallback/simplicity.
        Handles transitions INTO interaction modes.
        """
        logger.debug("Executing _run_nlu_pipeline (calling sequential steps)...")

        # 1. Classify Intent and Handle Clarification
        interaction_payload = self._classify_intent_and_handle_clarification(query)
        if interaction_payload:
            return (
                interaction_payload[0],
                interaction_payload[1],
            )  # Return interaction prompt

        # 2. Extract Parameters and Handle Validation
        interaction_payload = self._extract_parameters_and_handle_validation(query)
        if interaction_payload:
            return (
                interaction_payload[0],
                interaction_payload[1],
            )  # Return interaction prompt

        # 3. Generate Final Response
        final_result, final_hint = self._generate_final_response()
        return final_result, final_hint

    # --- Refactored Pipeline Steps ---

    def _classify_intent_and_handle_clarification(
        self, query: str
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """(Internal) Step 1: Classify intent and handle potential clarification.

        Updates self._pipeline_context with intent and confidence.
        If clarification is needed, sets interaction mode and returns interaction payload.
        Otherwise, returns None.
        """
        logger.debug("Pipeline Step 1: Classifying intent...")
        try:
            intent_result = self._intent_detector.classify_intent(
                query, self._pipeline_context
            )  # Pass context
            identified_intent = intent_result.get("intent", "unknown_intent")
            confidence = intent_result.get("confidence", 0.0)
            clarification_options = intent_result.get(
                "options"
            )  # Assume detector might return this

            logger.debug(
                f"Intent Classified: {identified_intent} (Confidence: {confidence:.2f})"
            )
            # Update context with initial intent/confidence
            self._pipeline_context.current_intent = identified_intent
            self._pipeline_context.confidence_score = confidence

        except Exception as e:
            logger.error(f"Error during Intent Classification: {e}", exc_info=True)
            # Raise or return error payload? Let's return error payload for run() to handle
            # This step shouldn't return the final error, maybe raise?
            # For now, log and let later steps fail or return error from run()
            # We need a consistent error handling strategy.
            # Let's return an error payload similar to interactions for now.
            return {
                "error": "Intent classification failed"
            }, "error"  # Return error tuple

        # --- Decision Point: Enter Clarification? ---
        clarification_threshold = 0.7  # TODO: Configurable
        if (
            confidence < clarification_threshold
            and identified_intent != "unknown_intent"
        ):
            logger.info(
                f"Confidence {confidence:.2f} below threshold {clarification_threshold}. Checking for clarification options."
            )
            if clarification_options:
                logger.info("Entering clarification mode.")
                from .nlu_pipeline.interaction_models import ClarificationData

                clar_data = ClarificationData(
                    user_input=query,  # Pass original query
                    options=clarification_options,
                    original_query=query,
                )
                self._pipeline_context.interaction_mode = (
                    InteractionState.CLARIFYING_INTENT
                )
                self._pipeline_context.interaction_data = clar_data

                handler = self._interaction_handlers[InteractionState.CLARIFYING_INTENT]
                prompt = handler.get_initial_prompt(self._pipeline_context)
                payload = {
                    "interaction_prompt": prompt,
                    "interaction_mode": InteractionState.CLARIFYING_INTENT,
                }
                hint = "awaiting_clarification"
                logger.debug(
                    f"Clarification needed. Returning interaction payload: {payload}, Hint: {hint}"
                )
                return payload, hint
            else:
                logger.warning(
                    "Clarification potentially needed, but no options provided. Proceeding with low confidence intent."
                )
                # Proceed to next step

        logger.debug(
            "Pipeline Step 1: Intent classification complete (no clarification needed)."
        )
        return None  # No interaction triggered

    def _extract_parameters_and_handle_validation(
        self, query: str
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """(Internal) Step 2: Extract parameters and handle potential validation.

        Uses intent from self._pipeline_context. Updates context with parameters.
        If validation is needed, sets interaction mode and returns interaction payload.
        Otherwise, returns None.
        """
        logger.debug("Pipeline Step 2: Extracting parameters...")
        identified_intent = self._pipeline_context.current_intent
        if not identified_intent or identified_intent == "unknown_intent":
            logger.debug("Skipping parameter extraction for unknown or missing intent.")
            self._pipeline_context.current_parameters = {}  # Ensure params are empty
            return None  # Nothing to extract or validate

        try:
            parameters, validation_requests = self._param_extractor.identify_parameters(
                query, identified_intent, self._pipeline_context
            )  # Pass context
            logger.debug(f"Parameters Extracted: {parameters}")
            logger.debug(f"Validation Requests: {validation_requests}")
            # Update context
            self._pipeline_context.current_parameters = parameters

        except Exception as e:
            logger.error(f"Error during Parameter Extraction: {e}", exc_info=True)
            # Return error payload
            return {"error": "Parameter extraction failed"}, "error"

        # --- Decision Point: Enter Validation? ---
        if validation_requests:
            validation_request = validation_requests[0]  # Handle one at a time
            # Break long log message
            log_message = (
                f"Validation required for parameter: {validation_request.parameter_name}. "
                f"Reason: {validation_request.reason}. Entering validation."
            )
            logger.info(log_message)
            from .nlu_pipeline.interaction_models import ValidationData

            # Construct ValidationData
            error_msg = f"Missing required parameter: {validation_request.parameter_name}"  # Basic message
            if validation_request.reason == "invalid_format":
                error_msg = f"Invalid format for {validation_request.parameter_name}. Please provide a valid value."
            # TODO: Improve error message generation based on param_meta type etc.

            val_data = ValidationData(
                user_input=query,  # Pass original query
                parameter_name=validation_request.parameter_name,
                error_message=error_msg,
                current_value=validation_request.current_value,  # Pass if available
            )
            self._pipeline_context.interaction_mode = (
                InteractionState.VALIDATING_PARAMETER
            )
            self._pipeline_context.interaction_data = val_data

            handler = self._interaction_handlers[InteractionState.VALIDATING_PARAMETER]
            prompt = handler.get_initial_prompt(self._pipeline_context)
            payload = {
                "interaction_prompt": prompt,
                "interaction_mode": InteractionState.VALIDATING_PARAMETER,
            }
            hint = "awaiting_validation"
            logger.debug(
                f"Validation needed. Returning interaction payload: {payload}, Hint: {hint}"
            )
            return payload, hint

        logger.debug(
            "Pipeline Step 2: Parameter extraction complete (no validation needed)."
        )
        return None  # No interaction triggered

    def _generate_final_response(self) -> Tuple[Dict[str, Any], str]:
        """(Internal) Step 3: Generate the final response text.

        Uses intent and parameters from self._pipeline_context.
        Returns the final NLU result payload and hint.
        """
        logger.debug("Pipeline Step 3: Generating final response...")
        identified_intent = self._pipeline_context.current_intent
        parameters = self._pipeline_context.current_parameters
        confidence = self._pipeline_context.confidence_score

        if not identified_intent or identified_intent == "unknown_intent":
            logger.debug("Generating response for unknown intent.")
            # Handle unknown intent response generation specifically
            # Maybe call text generator with a special intent key?
            # For now, return minimal payload
            return {
                "intent": "unknown_intent",
                "parameters": {},
                "confidence": confidence,
                "raw_response": None,
                "response_text": "Sorry, I didn't understand that.",  # Default unknown response
            }, "unknown_intent"

        try:
            # Revert: Pass context again, as required by the interface
            raw_response, response_text = self._text_generator.generate_response(
                identified_intent, parameters, self._pipeline_context
            )
            logger.debug(f"Generated Raw Response: {raw_response}")
            logger.debug(f"Generated Response Text: {response_text}")
        except Exception as e:
            logger.error(f"Error during Response Generation: {e}", exc_info=True)
            # Return error payload
            return {"error": "Response generation failed"}, "error"

        # Final successful NLU result
        nlu_result: Dict[str, Any] = {
            "intent": identified_intent,
            "parameters": parameters,
            "confidence": confidence,
            "raw_response": raw_response,
            "response_text": response_text,
        }
        hint = "new_conversation"  # Standard success hint

        # Potentially update internal history state here if needed
        # self._pipeline_context.conversation_history.append(...)? -> Context needs history attr

        logger.info(f"TalkEngine run() completed. Intent: {identified_intent}")
        logger.debug("Pipeline Step 3: Response generation complete.")
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
            self, user_input: str, context: NLUPipelineContext, excluded_intents=None
        ) -> Dict[str, Any]:
            logger.debug("MyIntentDetector classify running")
            if "search" in user_input:
                return {"intent": "search.web", "confidence": 0.99}
            return {"intent": "override_unknown", "confidence": 0.5}

    class MyParamExtractor(ParameterExtractionInterface):
        def __init__(self, config):
            logger.debug("MyParamExtractor initialized")

        def identify_parameters(
            self, user_input: str, intent: str, context: NLUPipelineContext
        ) -> Tuple[Dict[str, Any], List[Any]]:
            logger.debug("MyParamExtractor identify running")
            if intent == "search.web":
                return {"search_term": user_input.split("search for ")[-1]}, []
            return {"extracted_by": "override"}, []

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
