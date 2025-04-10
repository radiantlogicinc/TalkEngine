"""
Defines the core TalkEngine class.
"""

from typing import Any, Optional, Dict, List, Union, Callable, Mapping

# Assuming simplified types are defined or basic types used
# from .types import CommandMetadataInput, ConversationHistoryInput, NLUOverridesInput, NLURunResult
from .nlu_pipeline.nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    TextGenerationInterface,
)

# Import NLU models and interaction components - Corrected path
from .models import (
    NLUResult,
    ConversationDetail,
    InteractionLogEntry,
)
from .nlu_pipeline.models import (
    NLUPipelineContext,
    InteractionState,
    NLUPipelineState,
)
from .nlu_pipeline.interaction_handlers import (
    BaseInteractionHandler,
    ClarificationHandler,
    ValidationHandler,
)
from .nlu_pipeline.default_intent_detection import DefaultIntentDetection
from .nlu_pipeline.default_param_extraction import DefaultParameterExtraction
from .nlu_pipeline.default_text_generation import DefaultTextGeneration

# Corrected interaction model path
from .nlu_pipeline.interaction_models import (
    ValidationRequestInfo,
    ValidationData,
    ClarificationData,
)
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
    _nlu_overrides_config: Mapping[str, NLUImplementation]  # Store the input config
    _intent_detector: IntentDetectionInterface
    _param_extractor: ParameterExtractionInterface
    _text_generator: Optional[TextGenerationInterface]
    _is_trained: bool
    # Add pipeline context and interaction handlers
    _pipeline_context: NLUPipelineContext
    _interaction_handlers: Mapping[InteractionState, BaseInteractionHandler]

    def __init__(
        self,
        command_metadata: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        nlu_overrides: Optional[Mapping[str, NLUImplementation]] = None,
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
        nlu_overrides: Optional[Mapping[str, NLUImplementation]] = None,
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
        self._pipeline_context = NLUPipelineContext()
        # Store static config in context if needed by components
        # (Alternatively, pass directly during component calls)
        # self._pipeline_context.command_metadata = self._command_metadata
        # self._pipeline_context.conversation_history = self._conversation_history

        # Initialize NLU components (intent detection, parameter extraction, response generation)
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
        if "text_generation" in self._nlu_overrides_config:
            # Key provided, check if instance is valid
            if isinstance(text_gen_override, TextGenerationInterface):
                self._text_generator = text_gen_override
                logger.debug("Using provided TextGeneration override.")
            else:  # Key provided but invalid type
                logger.warning(
                    "Invalid TextGeneration override provided (type mismatch). Using default."
                )
                self._text_generator = DefaultTextGeneration()  # Use default
        else:
            # Key NOT provided, use default implementation
            logger.debug(
                "No TextGeneration override provided. Using DefaultTextGeneration."
            )
            self._text_generator = DefaultTextGeneration()

        logger.debug("NLU components initialized.")

    def _initialize_interaction_handlers(self):
        """(Internal) Initializes interaction handler instances."""
        logger.debug("Initializing Interaction Handlers...")
        self._interaction_handlers = {
            InteractionState.CLARIFYING_INTENT: ClarificationHandler(),
            InteractionState.VALIDATING_PARAMETER: ValidationHandler(),
            # InteractionState.AWAITING_FEEDBACK: FeedbackHandler(), # Removed - State not defined
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

    def run(
        self, user_query: str, excluded_intents: Optional[List[str]] = None
    ) -> NLUResult:
        """Processes a single natural language query.

        Handles interaction modes (clarification, validation) and the main NLU pipeline
        (intent, params, optional code execution, text generation).
        Logs interactions and returns a structured NLUResult.

        Args:
            user_query: The user's natural language input.
            excluded_intents: Optional list of intents to exclude from classification.

        Returns:
            An NLUResult object containing the outcome of the processing attempt.
        """
        logger.info(f"TalkEngine run() called with query: '{user_query}'")
        logger.debug(f"Initial context for run: {self._pipeline_context}")

        # --- Local State for this Run ---
        current_interactions: List[InteractionLogEntry] = []
        # Initialize based on existing interaction state if applicable
        initial_prompt_from_context: Optional[str] = None
        if self._pipeline_context.interaction_mode is not None:
            # Attempt to get prompt from existing interaction data
            if hasattr(self._pipeline_context.interaction_data, "prompt"):
                initial_prompt_from_context = getattr(
                    self._pipeline_context.interaction_data, "prompt", None
                )
            else:
                logger.warning(
                    "Interaction mode set, but interaction_data has no 'prompt' attribute."
                )

        current_last_prompt_shown: Optional[str] = (
            initial_prompt_from_context  # Use context prompt if available
        )
        # --- End Local State ---

        goto_step = NLUPipelineState.INTENT_CLASSIFICATION.value

        # --- FSM Logic (Handle if already in interaction) ---
        if self._pipeline_context.interaction_mode is not None:
            if handler := self._interaction_handlers.get(
                self._pipeline_context.interaction_mode
            ):
                logger.debug(
                    f"In interaction mode: {self._pipeline_context.interaction_mode}. Handling input."
                )
                # Log interaction before calling handler, using local prompt state
                if current_last_prompt_shown:
                    log_entry: InteractionLogEntry = (
                        self._pipeline_context.interaction_mode.value,
                        current_last_prompt_shown,  # Use local variable
                        user_query,
                    )
                    current_interactions.append(log_entry)  # Use local variable
                else:
                    logger.warning("Cannot log interaction: last_prompt_shown is None.")

                # Call handler with new signature
                (
                    updated_context,
                    proceed_flag,
                    goto_step_from_handler,
                    response_if_returning,
                ) = handler.handle_input(
                    self._pipeline_context, user_query  # Pass context first
                )
                self._pipeline_context = updated_context  # Update engine's context
                # Set local prompt variable based on what handler set (for potential immediate return) - REMOVED
                # current_last_prompt_shown = self._pipeline_context.last_prompt_shown

                # New logic based on tuple return
                if not proceed_flag:
                    # Handler decided not to proceed, return immediately
                    logger.debug("Interaction handler determined immediate return.")
                    # The handler should have set context.last_prompt_shown if returning a prompt - NO, use 4th element
                    return NLUResult(
                        command=self._pipeline_context.current_intent,  # May be partially set
                        parameters=self._pipeline_context.current_parameters,
                        artifacts=self._pipeline_context.artifacts,
                        conversation_detail=ConversationDetail(
                            interactions=current_interactions,
                            response_text=response_if_returning,  # Use response from handler
                        ),
                    )
                else:
                    # Handler decided to proceed, set the next step for the main pipeline
                    logger.debug(
                        f"Interaction handler proceeding to step: {goto_step_from_handler}"
                    )
                    # Map handler's goto step (potentially None) to engine's goto_step
                    # Use NLUPipelineState enum values for comparison/assignment
                    if (
                        goto_step_from_handler
                        == NLUPipelineState.PARAMETER_IDENTIFICATION.value
                    ):
                        goto_step = NLUPipelineState.PARAMETER_IDENTIFICATION.value
                    elif (
                        goto_step_from_handler == NLUPipelineState.CODE_EXECUTION.value
                    ):
                        goto_step = NLUPipelineState.CODE_EXECUTION.value
                    # Add more specific steps if needed
                    else:
                        # Default if handler doesn't specify or provides unknown step
                        logger.warning(
                            f"Handler returned goto_step='{goto_step_from_handler}', defaulting to intent classification."
                        )
                        goto_step = NLUPipelineState.INTENT_CLASSIFICATION.value
            else:
                logger.error(
                    f"No handler for mode {self._pipeline_context.interaction_mode}. Resetting."
                )
                self._pipeline_context.interaction_mode = None
                self._pipeline_context.interaction_data = None
                goto_step = (
                    NLUPipelineState.INTENT_CLASSIFICATION.value
                )  # Ensure goto_step is set

        # --- Main NLU Pipeline ---
        logger.debug(f"Running main NLU pipeline starting from step: {goto_step}")
        text_response: Optional[str] = None  # Initialize text_response
        code_exec_result: Optional[Dict[str, Any]] = None  # Initialize code_exec_result

        if goto_step == NLUPipelineState.INTENT_CLASSIFICATION.value:
            logger.debug("Running main NLU pipeline: Step 1 - Intent Classification")
            # 1. Intent Classification
            try:
                intent_result = self._intent_detector.classify_intent(
                    user_query, self._pipeline_context, excluded_intents
                )
                # Assign intent and confidence score from result to context
                self._pipeline_context.current_intent = intent_result.get(
                    "intent", "unknown"
                )
                self._pipeline_context.confidence_score = intent_result.get(
                    "confidence", 0.0  # Default to 0.0 if not provided
                )
                logger.debug(
                    f"Intent: {self._pipeline_context.current_intent}, Confidence: {self._pipeline_context.confidence_score}"
                )
            except Exception:
                logger.exception("Error during intent classification.")
                self._pipeline_context.current_intent = "unknown"
                # TODO: Decide if we should return error NLUResult here?
                # For now, continue with unknown intent.

            # 1.a. Decision Point: Intent Clarification
            clarification_threshold = 0.6
            if (
                self._pipeline_context.current_intent != "unknown"
                # Check confidence is not None before comparing
                and self._pipeline_context.confidence_score is not None
                and self._pipeline_context.confidence_score < clarification_threshold
            ):
                logger.info(
                    f"Intent confidence low ({self._pipeline_context.confidence_score:.2f}). Entering clarification."
                )
                # TODO: Get clarification options (e.g., top N intents from detector?)
                # Ensure current_intent is not None before adding
                clarification_options = []
                if self._pipeline_context.current_intent:
                    clarification_options.append(self._pipeline_context.current_intent)
                clarification_options.append("other_intent_example")  # Placeholder

                clar_data = ClarificationData(
                    prompt="Which command did you mean?", options=clarification_options
                )
                self._pipeline_context.interaction_mode = (
                    InteractionState.CLARIFYING_INTENT
                )
                self._pipeline_context.interaction_data = clar_data
                if handler := self._interaction_handlers.get(
                    InteractionState.CLARIFYING_INTENT
                ):
                    try:
                        # Pass local state to helper
                        return self._extracted_from_run_(
                            handler, current_interactions, current_last_prompt_shown
                        )
                    except Exception:
                        self._extracted_from_run_194(
                            "Error getting clarification prompt. Aborting clarification."
                        )
                else:
                    logger.error("Clarification handler not found!")
                    self._pipeline_context.interaction_mode = None  # Reset state

            # If clarification wasn't entered, proceed to next step flag
            goto_step = NLUPipelineState.PARAMETER_IDENTIFICATION.value

        if goto_step == NLUPipelineState.PARAMETER_IDENTIFICATION.value:
            logger.debug("Running main NLU pipeline: Step 2 - Parameter Extraction")
            # 2. Parameter Extraction (only if intent is known)
            validation_requests: List[ValidationRequestInfo] = []
            # Preserve params if proceeding immediately from intent clarification
            # params_before_extraction = self._pipeline_context.current_parameters.copy()

            if (
                self._pipeline_context.current_intent
                and self._pipeline_context.current_intent != "unknown"
            ):
                try:
                    # Pass user_query, but extractor should rely on context.current_intent
                    extracted_params, validation_requests = (
                        self._param_extractor.identify_parameters(
                            user_query,  # This might be the interaction response!
                            self._pipeline_context.current_intent,
                            self._pipeline_context,
                        )
                    )
                    # If proceeding from intent clarification, merge/update params?
                    # No, context should already be updated by handler. Replace params.
                    # if self._pipeline_context.interaction_mode != InteractionState.IDLE:
                    #      # If we are here after proceed_immediately, context should be updated.
                    #      # Don't overwrite params set by interaction handler?
                    #      pass # Trust context params set by handler
                    # else:
                    # If running fresh, set extracted params
                    self._pipeline_context.current_parameters = extracted_params

                    logger.debug(
                        f"Parameters updated: {self._pipeline_context.current_parameters}"
                    )
                    logger.debug(
                        f"Validation requests generated: {validation_requests}"
                    )
                except Exception:
                    logger.exception("Error during parameter extraction.")
                    # Restore params if proceeding from interaction?
                    # if self._pipeline_context.interaction_mode != InteractionState.IDLE:
                    #    self._pipeline_context.current_parameters = params_before_extraction
                    # else:
                    # If running fresh and extraction fails, clear params
                    self._pipeline_context.current_parameters = {}
                    # Continue, but parameters might be missing

            # 2.a. Decision Point: Parameter Validation
            if validation_requests:
                request = validation_requests[0]
                logger.info(
                    f"Required parameter '{request.parameter_name}' missing/invalid ({request.reason}). Entering validation."
                )
                val_data = ValidationData(
                    parameter_name=request.parameter_name,
                    reason=request.reason,
                    current_value=request.current_value,
                    prompt=f"What value should I use for '{request.parameter_name}'?",  # Placeholder
                )
                self._pipeline_context.interaction_mode = (
                    InteractionState.VALIDATING_PARAMETER
                )
                self._pipeline_context.interaction_data = val_data
                if handler := self._interaction_handlers.get(
                    InteractionState.VALIDATING_PARAMETER
                ):
                    try:
                        # Pass local state to helper
                        return self._extracted_from_run_256(
                            handler, current_interactions, current_last_prompt_shown
                        )
                    except Exception:
                        self._extracted_from_run_194(
                            "Error during validation prompt generation/return. Aborting validation."
                        )
                else:
                    logger.error("Validation handler not found!")
                    self._pipeline_context.interaction_mode = None

            goto_step = NLUPipelineState.CODE_EXECUTION.value

        if goto_step == NLUPipelineState.CODE_EXECUTION.value:
            logger.debug("Running main NLU pipeline: Step 3 - Code Execution")
            # 3. Code Execution (Optional)
            if (
                self._pipeline_context.current_intent
                and self._pipeline_context.current_intent != "unknown"
            ):
                intent_meta = self._command_metadata.get(
                    self._pipeline_context.current_intent, {}
                )
                executable_code: Optional[Callable] = intent_meta.get("executable_code")

                if callable(executable_code):
                    logger.debug(
                        f"Found executable code for intent: {self._pipeline_context.current_intent}"
                    )
                    try:
                        # Pass only parameters for now, could pass context if needed
                        code_exec_result = executable_code(
                            self._pipeline_context.current_parameters
                        )
                        logger.info(
                            f"Code execution successful. Result: {code_exec_result}"
                        )
                    except Exception as e:
                        logger.exception(
                            f"Error executing code for intent {self._pipeline_context.current_intent}"
                        )
                        code_exec_result = {
                            "error": f"Execution failed: {e}"
                        }  # Store error in result
                else:
                    logger.debug(
                        f"No executable code found or configured for intent: {self._pipeline_context.current_intent}"
                    )

            # Proceed to next step flag
            goto_step = NLUPipelineState.RESPONSE_TEXT_GENERATION.value

        if goto_step == NLUPipelineState.RESPONSE_TEXT_GENERATION.value:
            logger.debug("Running main NLU pipeline: Step 4 - Text Generation")
            # 4. Text Generation (Mandatory)
            text_response = ""
            # Generate text even for unknown intent to provide feedback
            intent_to_generate = self._pipeline_context.current_intent or "unknown"
            params_to_generate = self._pipeline_context.current_parameters

            try:
                if self._text_generator is not None:
                    generated_text = self._text_generator.generate_text(
                        intent_to_generate,
                        params_to_generate,
                        code_exec_result,  # Pass code result (might be None)
                        self._pipeline_context,
                    )
                    if generated_text is not None:
                        text_response = generated_text
                        logger.debug(f"Generated text response: {text_response}")
                    else:
                        text_response = (
                            "Sorry, I could not generate a specific response for that."
                        )

                        logger.warning(
                            "Text generator returned None. Using fallback response."
                        )
                else:
                    logger.warning(
                        "Text generator is not configured. Cannot generate response."
                    )
                    text_response = (
                        "Sorry, the text generation component is not available."
                    )
            except Exception:
                logger.exception("Error during text generation.")
                # Ensure text_response is assigned even after exception
                text_response = "Sorry, I encountered an error generating a response."

        # 5. Final Result Construction
        final_conv_detail = ConversationDetail(
            interactions=current_interactions,  # Use local variable
            response_text=text_response,
        )
        final_nlu_result = NLUResult(
            command=self._pipeline_context.current_intent,
            parameters=self._pipeline_context.current_parameters,
            artifacts=code_exec_result,
            conversation_detail=final_conv_detail,
        )

        logger.info(f"TalkEngine run() completed. Result: {final_nlu_result}")
        return final_nlu_result

    # TODO Rename this here and in `run`
    def _extracted_from_run_256(
        self,
        handler,
        current_interactions_list: List[InteractionLogEntry],
        last_prompt: Optional[str],
    ):
        # This helper seems to handle returning mid-pipeline for validation
        # It needs access to the local state if it modifies/reads it.
        # For now, assume it only needs the context passed to the handler.
        # If it needs current_interactions/current_last_prompt_shown, they need passing.
        # Minimal change: Assume it works as is, relying on context state *before* this call.
        # Let's add the local state pass-through for consistency if needed.

        # This helper is called from within the main pipeline logic block,
        # Need to ensure local state `current_interactions` and `current_last_prompt_shown` are handled correctly.

        validation_prompt = handler.get_initial_prompt(self._pipeline_context)
        validation_log_entry: InteractionLogEntry = (
            InteractionState.VALIDATING_PARAMETER.value,
            validation_prompt,
            None,
        )
        # Append to the passed-in local list
        current_interactions_list.append(validation_log_entry)

        # The prompt to return *is* the validation prompt itself
        # self._pipeline_context.last_prompt_shown = validation_prompt # No! Use local var logic
        conv_detail = ConversationDetail(
            interactions=current_interactions_list,  # Use local list
            response_text=validation_prompt,  # Return the generated prompt
        )
        logger.debug(
            "Validation triggered. Returning NLUResult with prompt."
        )  # Added logging
        return NLUResult(
            command=self._pipeline_context.current_intent,
            parameters=self._pipeline_context.current_parameters,
            artifacts=None,
            conversation_detail=conv_detail,
        )

    # TODO Rename this here and in `run`
    def _extracted_from_run_(
        self,
        handler,
        current_interactions_list: List[InteractionLogEntry],
        last_prompt: Optional[str],
    ):
        # Similar issue as _extracted_from_run_256 regarding local state.

        clarification_prompt = handler.get_initial_prompt(self._pipeline_context)
        # Log interaction start
        clarification_log_entry: InteractionLogEntry = (
            InteractionState.CLARIFYING_INTENT.value,
            clarification_prompt,
            None,
        )
        # Append to the passed-in local list
        current_interactions_list.append(clarification_log_entry)

        # The prompt to return *is* the clarification prompt itself
        # self._pipeline_context.last_prompt_shown = clarification_prompt # No! Use local var logic

        # Return NLUResult indicating clarification needed
        conv_detail = ConversationDetail(
            interactions=current_interactions_list,  # Use local list
            response_text=clarification_prompt,  # Return the generated prompt
        )
        return NLUResult(
            command=None,  # Intent is not confirmed yet
            parameters={},
            artifacts=None,
            conversation_detail=conv_detail,
        )

    # TODO Rename this here and in `run`
    def _extracted_from_run_194(self, arg0):
        logger.exception(arg0)
        # These still modify context directly, which is fine as they exist.
        self._pipeline_context.interaction_mode = None
        self._pipeline_context.interaction_data = None

    # TODO Rename this here and in `run`
    def _extracted_from_run_73(
        self,
        arg0,
        interaction_result,
        current_interactions_list: List[InteractionLogEntry],
    ):
        logger.debug(arg0)
        conv_detail = ConversationDetail(
            interactions=current_interactions_list,  # Use passed-in local list
            response_text=interaction_result.response,
        )
        return NLUResult(
            command=self._pipeline_context.current_intent,
            parameters=self._pipeline_context.current_parameters,
            artifacts=None,
            conversation_detail=conv_detail,
        )

    def reset(
        self,
        command_metadata: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        nlu_overrides: Optional[Mapping[str, NLUImplementation]] = None,
    ) -> None:
        """Resets and re-initializes the engine with new data."""
        logger.info("Resetting TalkEngine...")
        # Re-run initialization logic using the helper method
        self._do_initialize(command_metadata, conversation_history, nlu_overrides)
