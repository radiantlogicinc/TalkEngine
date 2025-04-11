"""
Defines the core TalkEngine class.
"""

import inspect
from typing import Any, Optional, Callable, Mapping, Type

from pydantic import BaseModel, ValidationError

from .nlu_pipeline.nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    TextGenerationInterface,
)

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

from .nlu_pipeline.interaction_models import (
    ValidationData,
    ClarificationData,
)
from .utils.logging import logger

# Import local types and interfaces
from .types import (
    CommandMetadataConfig,
    NLUOverridesConfig,
)  # Add imports for new types


# Type hint for NLU override values (instances implementing interfaces)
NLUImplementation = (
    IntentDetectionInterface | ParameterExtractionInterface | TextGenerationInterface
)


# Removed type aliases moved to types.py
# ExecutableCodeOverride = Dict[str, Callable | Type[BaseModel]] # 'function', 'result_class'
# CommandOverride = Dict[str, ExecutableCodeOverride] # 'executable_code'
# NLUOverridesConfig = Dict[str, NLUImplementation | CommandOverride]
# CommandMetadataConfig = Dict[str, Dict[str, str | Type[BaseModel]]] # 'description', 'parameter_class'


class TalkEngine:
    """NLU Pipeline Engine.

    Processes natural language queries based on developer-provided command metadata
    and optional NLU overrides and conversation history.
    """

    # Use imported types
    _command_metadata: CommandMetadataConfig
    _conversation_history: list[dict[str, Any]]
    _nlu_overrides_config: NLUOverridesConfig
    _intent_detector: IntentDetectionInterface
    _param_extractor: ParameterExtractionInterface
    _text_generator: Optional[TextGenerationInterface]
    _is_trained: bool
    # Add pipeline context and interaction handlers
    _pipeline_context: NLUPipelineContext
    _interaction_handlers: Mapping[InteractionState, BaseInteractionHandler]

    def __init__(
        self,
        command_metadata: CommandMetadataConfig,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        nlu_overrides: Optional[NLUOverridesConfig] = None,
    ):
        """Initializes the TalkEngine with configuration.

        Args:
            command_metadata: A dictionary describing available commands.
                              Key: command name (str).
                              Value: dict with 'description' (str) and 'parameter_class' (Type[BaseModel]).
            conversation_history: An optional list of previous conversation turns.
                                  Example: [{ "role": "user", "content": "..." }, ...]
            nlu_overrides: Optional dictionary for overriding NLU components or providing
                           command execution logic.
                           Keys can be 'intent_detection', 'param_extraction', 'text_generation' mapping to
                           instances implementing the respective interfaces.
                           Keys can also be command names (str) mapping to a dict like:
                           { "executable_code": { "function": Callable, "result_class": Type[BaseModel] } }
        """
        # Use a helper method for core initialization logic
        self._do_initialize(command_metadata, conversation_history, nlu_overrides)

    def _do_initialize(
        self,
        command_metadata: CommandMetadataConfig,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        nlu_overrides: Optional[NLUOverridesConfig] = None,
    ):
        """Initializes or re-initializes the engine components."""
        # --- Validation --- (Moved validation before assignment)
        if not isinstance(command_metadata, dict):
            raise TypeError("command_metadata must be a dictionary.")
        self._validate_command_metadata(command_metadata)

        if conversation_history is not None and not isinstance(
            conversation_history, list
        ):
            raise TypeError("conversation_history must be a list if provided.")

        parsed_nlu_overrides = nlu_overrides or {}
        if not isinstance(parsed_nlu_overrides, dict):
            raise TypeError("nlu_overrides must be a dictionary if provided.")
        self._validate_nlu_overrides(parsed_nlu_overrides)
        # --- End Validation ---

        self._command_metadata = command_metadata
        self._conversation_history = conversation_history or []
        self._nlu_overrides_config = parsed_nlu_overrides

        # Initialize pipeline context (ensure context has access to validated metadata)
        self._pipeline_context = NLUPipelineContext(
            command_metadata=self._command_metadata
        )

        # Initialize NLU components (intent detection, parameter extraction, response generation)
        # using defaults or overrides
        self._initialize_nlu_components()

        # Initialize Interaction Handlers
        self._initialize_interaction_handlers()

        # Set trained status - assume not trained until train() is explicitly called
        self._is_trained = False

    def _validate_command_metadata(self, metadata: CommandMetadataConfig):
        """Validates the structure and types within command_metadata."""
        logger.debug("Validating command_metadata...")
        for cmd_name, cmd_def in metadata.items():
            if not isinstance(cmd_name, str):
                raise TypeError(
                    f"Command metadata keys must be strings, found: {cmd_name}"
                )
            if not isinstance(cmd_def, dict):
                raise TypeError(f"Command definition for '{cmd_name}' must be a dict.")
            if "description" not in cmd_def or not isinstance(
                cmd_def["description"], str
            ):
                raise ValueError(
                    f"Command '{cmd_name}' metadata missing string 'description'."
                )
            if "parameter_class" not in cmd_def:
                raise ValueError(
                    f"Command '{cmd_name}' metadata missing 'parameter_class'."
                )

            param_class = cmd_def["parameter_class"]
            # Use inspect.isclass() for safer type checking before issubclass
            if not inspect.isclass(param_class):
                raise TypeError(
                    f"'parameter_class' for command '{cmd_name}' must be a class type, got {type(param_class).__name__}."
                )
            if not issubclass(param_class, BaseModel):
                raise TypeError(
                    f"'parameter_class' for command '{cmd_name}' must be a subclass of pydantic.BaseModel."
                )
        logger.debug("command_metadata validation successful.")

    def _validate_nlu_overrides(self, overrides: NLUOverridesConfig):
        """Validates the structure and types within nlu_overrides."""
        logger.debug("Validating nlu_overrides...")
        nlu_interface_keys = {"intent_detection", "param_extraction", "text_generation"}
        valid_interfaces = {
            "intent_detection": IntentDetectionInterface,
            "param_extraction": ParameterExtractionInterface,
            "text_generation": TextGenerationInterface,
        }

        for key, value in overrides.items():
            if key in nlu_interface_keys:
                # Check NLU component overrides
                expected_interface = valid_interfaces[key]
                if not isinstance(value, expected_interface):
                    raise TypeError(
                        f"Override for '{key}' must implement {expected_interface.__name__}."
                    )
            else:
                # Assume it's a command execution override
                if not isinstance(key, str):
                    raise TypeError(
                        f"Command override key must be a string, found: {key}"
                    )
                if not isinstance(value, dict) or "executable_code" not in value:
                    raise ValueError(
                        f"Command override for '{key}' must be a dict containing 'executable_code'."
                    )

                exec_code_def = value["executable_code"]
                if not isinstance(exec_code_def, dict):
                    raise TypeError(
                        f"'executable_code' for command '{key}' override must be a dict."
                    )

                if "function" not in exec_code_def or not callable(
                    exec_code_def["function"]
                ):
                    raise ValueError(
                        f"'executable_code' for command '{key}' override must include a callable 'function'."
                    )

                if "result_class" not in exec_code_def:
                    raise ValueError(
                        f"'executable_code' for command '{key}' override must include 'result_class'."
                    )

                result_class = exec_code_def["result_class"]
                if not inspect.isclass(result_class):
                    raise TypeError(
                        f"'result_class' for command '{key}' override must be a class type"
                        f", got {type(result_class).__name__}."
                    )
                if not issubclass(result_class, BaseModel):
                    raise TypeError(
                        f"'result_class' for command '{key}' override must be a subclass of pydantic.BaseModel."
                    )
        logger.debug("nlu_overrides validation successful.")

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
        self, user_query: str, excluded_intents: Optional[list[str]] = None
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
        current_interactions: list[InteractionLogEntry] = []
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

        goto_step: Optional[str] = NLUPipelineState.INTENT_CLASSIFICATION.value

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
                    # Ensure we use the NamedTuple constructor
                    log_entry = InteractionLogEntry(
                        stage=self._pipeline_context.interaction_mode.value,
                        prompt=current_last_prompt_shown,  # Use local variable
                        response=user_query,
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

        # --- Main Pipeline Execution --- (If not handled by interaction mode above)
        final_result: Optional[NLUResult] = None
        try:
            # Reset context fields specific to a single run, *unless* we just came from interaction handler
            # If goto_step is already set by handler, don't reset these
            if goto_step == NLUPipelineState.INTENT_CLASSIFICATION.value:
                self._pipeline_context.artifacts = None
                self._pipeline_context.current_intent = None
                self._pipeline_context.current_parameters = {}
                self._pipeline_context.parameter_validation_errors = []
                self._pipeline_context.confidence_score = 0.0
            # Initialize ConversationDetail for this run, including interactions logged *before* the main loop
            current_conversation_detail: ConversationDetail = ConversationDetail(
                interactions=current_interactions
            )

            while goto_step:
                logger.debug(f"Processing step: {goto_step}")
                self._pipeline_context.current_state = NLUPipelineState(goto_step)
                next_step = None  # Default: finish if no specific next step

                # --- 1. Intent Classification ---
                if goto_step == NLUPipelineState.INTENT_CLASSIFICATION.value:
                    intent_result = self._intent_detector.classify_intent(
                        user_input=user_query,
                        context=self._pipeline_context,  # Pass context which has metadata
                        excluded_intents=excluded_intents,
                    )
                    self._pipeline_context.current_intent = intent_result.get("intent")
                    self._pipeline_context.confidence_score = intent_result.get(
                        "confidence", 0.0
                    )
                    logger.info(
                        f"Intent classification result: {self._pipeline_context.current_intent} "
                        f"(Confidence: {self._pipeline_context.confidence_score:.2f})"
                    )

                    # --- TODO: Intent Clarification Check ---
                    # Threshold below which clarification is triggered
                    CLARIFICATION_CONFIDENCE_THRESHOLD = 0.6
                    if (
                        self._pipeline_context.confidence_score
                        < CLARIFICATION_CONFIDENCE_THRESHOLD
                    ):
                        logger.info(
                            f"Intent confidence ({self._pipeline_context.confidence_score:.2f}) "
                            f"is below threshold ({CLARIFICATION_CONFIDENCE_THRESHOLD}). Entering clarification."
                        )
                        # Prepare interaction data (need potential options - TODO: get from intent detector?)
                        # For now, use a placeholder - Handler should ideally get options
                        # Assume handler can get options or use the current intent as one option
                        # Here we assume the handler needs the low-confidence intent to start
                        clarification_data = ClarificationData(
                            prompt="",
                            options=[
                                self._pipeline_context.current_intent or "unknown"
                            ],
                        )  # Handler sets real prompt/options
                        self._pipeline_context.interaction_mode = (
                            InteractionState.CLARIFYING_INTENT
                        )
                        self._pipeline_context.interaction_data = clarification_data

                        if handler := self._interaction_handlers.get(
                            InteractionState.CLARIFYING_INTENT
                        ):
                            prompt = handler.get_initial_prompt(self._pipeline_context)
                            current_last_prompt_shown = prompt
                            current_interactions.append(
                                InteractionLogEntry(
                                    stage=InteractionState.CLARIFYING_INTENT.value,
                                    prompt=prompt,
                                    response=None,
                                )
                            )
                            # Return intermediate result indicating clarification needed
                            final_result = NLUResult(
                                command=None,  # No command confirmed yet
                                parameters={},
                                artifacts=None,
                                conversation_detail=ConversationDetail(
                                    interactions=current_interactions,
                                    response_text=prompt,  # Use prompt as response
                                ),
                            )
                            logger.info(f"Entering clarification interaction: {prompt}")
                            next_step = None  # Exit pipeline for interaction
                            break  # Exit the while loop
                        else:
                            logger.error(
                                "Clarification needed but no handler found! Proceeding with low confidence."
                            )
                            # If no handler, just proceed to param extraction with the low-confidence intent
                            next_step = NLUPipelineState.PARAMETER_IDENTIFICATION
                    # elif self._pipeline_context.current_intent == "needs_clarification":
                    #    # Set interaction mode, data, handler, get prompt
                    #    # ... logic to prepare clarification ...

                    if (
                        not self._pipeline_context.current_intent
                        or self._pipeline_context.current_intent == "unknown"
                    ):
                        logger.warning(
                            "Could not determine intent or intent is unknown."
                        )
                        # Decide if we proceed to text generation for unknown intent or stop
                        next_step = NLUPipelineState.RESPONSE_TEXT_GENERATION
                    else:
                        next_step = NLUPipelineState.PARAMETER_IDENTIFICATION

                # --- 2. Parameter Extraction ---
                elif goto_step == NLUPipelineState.PARAMETER_IDENTIFICATION.value:
                    # Requires a valid intent from the previous step
                    if (
                        not self._pipeline_context.current_intent
                        or self._pipeline_context.current_intent == "unknown"
                    ):
                        logger.error(
                            "Parameter extraction called without a valid intent."
                        )
                        # Skip to response generation or handle error
                        next_step = NLUPipelineState.RESPONSE_TEXT_GENERATION
                    else:
                        try:
                            # --- Retrieve parameter_class for the intent ---
                            command_def = self._command_metadata.get(
                                self._pipeline_context.current_intent
                            )
                            if not command_def or "parameter_class" not in command_def:
                                logger.error(
                                    f"Metadata missing or invalid for intent "
                                    f"'{self._pipeline_context.current_intent}'. Cannot extract parameters."
                                )
                                next_step = NLUPipelineState.RESPONSE_TEXT_GENERATION
                            else:
                                # Assure MyPy that parameter_class is the correct type here
                                param_class_val = command_def["parameter_class"]
                                assert isinstance(param_class_val, type) and issubclass(
                                    param_class_val, BaseModel
                                )
                                parameter_class: Type[BaseModel] = param_class_val

                                # --- Call parameter extractor --- (Updated Call)
                                extracted_params, validation_requests = (
                                    self._param_extractor.identify_parameters(
                                        user_input=user_query,
                                        intent=self._pipeline_context.current_intent,
                                        parameter_class=parameter_class,  # Pass the class
                                        context=self._pipeline_context,
                                    )
                                )
                                self._pipeline_context.current_parameters = (
                                    extracted_params
                                )
                                self._pipeline_context.parameter_validation_errors = [
                                    f"{req.parameter_name}: {req.reason}"
                                    for req in validation_requests
                                ]  # Store simple error strings for now
                                logger.info(
                                    f"Parameter extraction result: {extracted_params}"
                                )
                                logger.debug(
                                    f"Validation requests: {validation_requests}"
                                )

                                # --- Parameter Validation Check ---
                                if validation_requests:
                                    # 1. Create ValidationData (without prompt) and set context
                                    validation_data = ValidationData(
                                        requests=validation_requests, prompt=None
                                    )
                                    self._pipeline_context.interaction_mode = (
                                        InteractionState.VALIDATING_PARAMETER
                                    )
                                    self._pipeline_context.interaction_data = (
                                        validation_data
                                    )

                                    # 2. Get the specific handler
                                    if handler := self._interaction_handlers.get(
                                        InteractionState.VALIDATING_PARAMETER
                                    ):
                                        # 3. Get the prompt from the handler
                                        prompt = handler.get_initial_prompt(
                                            self._pipeline_context
                                        )
                                        # 4. Store the prompt back into the context data
                                        if isinstance(
                                            self._pipeline_context.interaction_data,
                                            ValidationData,
                                        ):
                                            self._pipeline_context.interaction_data.prompt = (
                                                prompt
                                            )
                                        else:
                                            logger.warning(
                                                "Interaction data is not ValidationData type, cannot store prompt."
                                            )

                                        # 5. Log and prepare intermediate result
                                        current_last_prompt_shown = (
                                            prompt  # Update local state
                                        )
                                        current_interactions.append(
                                            InteractionLogEntry(
                                                stage=InteractionState.VALIDATING_PARAMETER.value,
                                                prompt=prompt,
                                                response=None,
                                            )
                                        )
                                        final_result = NLUResult(
                                            command=self._pipeline_context.current_intent,
                                            parameters=self._pipeline_context.current_parameters,
                                            artifacts=None,
                                            conversation_detail=ConversationDetail(
                                                interactions=current_interactions,
                                                response_text=prompt,  # Use prompt as response
                                            ),
                                        )
                                        logger.info(
                                            f"Entering validation interaction: {prompt}"
                                        )
                                        next_step = (
                                            None  # Exit pipeline for interaction
                                        )
                                        break  # Exit the while loop
                                    else:
                                        logger.error(
                                            "Validation needed but no handler found!"
                                        )
                                        next_step = (
                                            NLUPipelineState.RESPONSE_TEXT_GENERATION
                                        )
                                else:
                                    # Parameters extracted successfully, proceed to code execution
                                    next_step = NLUPipelineState.CODE_EXECUTION

                        except Exception as e:
                            logger.error(
                                f"Error during parameter extraction: {e}", exc_info=True
                            )
                            next_step = NLUPipelineState.RESPONSE_TEXT_GENERATION

                # --- 3. Code Execution --- (Rewritten Logic)
                elif goto_step == NLUPipelineState.CODE_EXECUTION.value:
                    intent = self._pipeline_context.current_intent
                    self._pipeline_context.artifacts = (
                        None  # Ensure artifacts are None initially
                    )

                    # Check if overrides exist and contain executable code for this intent
                    if (
                        intent
                        and self._nlu_overrides_config
                        and intent in self._nlu_overrides_config
                    ):
                        command_override = self._nlu_overrides_config[intent]
                        if (
                            isinstance(command_override, dict)
                            and "executable_code" in command_override
                        ):
                            exec_code_def = command_override["executable_code"]

                            # Double check structure (validated in init, but safer here)
                            if (
                                isinstance(exec_code_def, dict)
                                and "function" in exec_code_def
                                and callable(exec_code_def["function"])
                                and "result_class" in exec_code_def
                                and inspect.isclass(exec_code_def["result_class"])
                                and issubclass(exec_code_def["result_class"], BaseModel)
                            ):  # Added callable check

                                function_to_run: Callable = exec_code_def["function"]
                                result_class: Type[BaseModel] = exec_code_def[
                                    "result_class"
                                ]
                                # parameter_class should already be defined and validated from PARAMETER_IDENTIFICATION step
                                # No need to redefine it here.
                                # parameter_class: Type[BaseModel] = self._command_metadata[intent]["parameter_class"]

                                logger.info(
                                    f"Attempting to execute function for command '{intent}'"
                                )
                                try:
                                    # 1. Instantiate parameters using extracted dict
                                    param_object = parameter_class(
                                        **self._pipeline_context.current_parameters
                                    )
                                    logger.debug(
                                        f"Instantiated parameter object: {param_object}"
                                    )

                                    # 2. Run function
                                    result_object = function_to_run(param_object)
                                    logger.debug(f"Function returned: {result_object}")

                                    # 3. Validate result type
                                    if not isinstance(result_object, result_class):
                                        logger.error(
                                            f"Executable function for '{intent}' returned type "
                                            f"{type(result_object).__name__}, expected {result_class.__name__}"
                                        )
                                        # Handle error - artifacts remain None
                                    else:
                                        # 4. Store valid result
                                        self._pipeline_context.artifacts = result_object
                                        logger.info(
                                            f"Stored execution artifacts: {self._pipeline_context.artifacts}"
                                        )

                                except ValidationError as e:
                                    logger.error(
                                        f"Pydantic validation error instantiating {parameter_class.__name__} "
                                        f"for '{intent}': {e}"
                                    )
                                    # artifacts remain None
                                except Exception as e:
                                    logger.error(
                                        f"Error executing function for command '{intent}': {e}",
                                        exc_info=True,
                                    )
                                    # artifacts remain None
                            else:
                                logger.warning(
                                    f"Executable code definition for '{intent}' in overrides is malformed."
                                )
                        else:
                            logger.debug(
                                f"No 'executable_code' defined in override for command '{intent}'."
                            )
                    else:
                        logger.debug(
                            f"No overrides or no executable_code found for command '{intent}'."
                        )

                    # Always proceed to response generation after attempting execution
                    next_step = NLUPipelineState.RESPONSE_TEXT_GENERATION

                # --- 4. Response Text Generation ---
                elif goto_step == NLUPipelineState.RESPONSE_TEXT_GENERATION.value:
                    text_response = None
                    if self._text_generator:
                        try:
                            # Call generator, passing the BaseModel artifact if it exists (Updated call)
                            text_response = self._text_generator.generate_text(
                                command=self._pipeline_context.current_intent
                                or "unknown",
                                parameters=self._pipeline_context.current_parameters,
                                artifacts=self._pipeline_context.artifacts,  # Pass Optional[BaseModel]
                                context=self._pipeline_context,
                            )
                            logger.info(f"Generated text response: {text_response}")
                        except Exception as e:
                            logger.error(
                                f"Error during text generation: {e}", exc_info=True
                            )
                            text_response = (
                                "Sorry, I encountered an error generating a response."
                            )
                    else:
                        logger.warning("No text generator configured.")

                    # Store the final text response in the conversation detail for this run
                    # Revert to direct assignment; goto_step hint should fix MyPy
                    if text_response is not None:
                        current_conversation_detail.response_text = text_response
                    else:
                        current_conversation_detail.response_text = None

                    # This is the last step in the normal flow
                    next_step = None
                else:
                    logger.error(f"Reached unknown pipeline step: {goto_step}")
                    next_step = None  # Exit loop on error

                # --- Update Loop Control ---
                goto_step = next_step.value if next_step else None

            # --- End of Pipeline Execution (Normal Flow) ---
            # Construct the final result if not already set by interaction break
            if final_result is None:
                final_result = NLUResult(
                    command=self._pipeline_context.current_intent,
                    parameters=self._pipeline_context.current_parameters,
                    artifacts=self._pipeline_context.artifacts,
                    conversation_detail=current_conversation_detail,  # Use detail built during this run
                )

        except Exception as e:
            # Catch-all for unexpected errors during pipeline execution
            logger.error(
                f"Unexpected error during TalkEngine.run pipeline: {e}", exc_info=True
            )
            # Construct an error result
            error_detail = ConversationDetail(
                interactions=current_interactions,
                response_text="Sorry, an unexpected error occurred.",
            )
            final_result = NLUResult(
                command="unknown", conversation_detail=error_detail
            )

        # --- Final Logging and Return ---
        logger.debug(f"Final NLU Result: {final_result}")
        logger.info("TalkEngine run() finished.")
        return final_result  # Return the single NLUResult object

    def reset(
        self,
        command_metadata: CommandMetadataConfig,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        nlu_overrides: Optional[NLUOverridesConfig] = None,
    ) -> None:
        """Re-initializes the engine with new configuration.

        Args:
            command_metadata: New command metadata (structure as in __init__).
            conversation_history: New conversation history (optional).
            nlu_overrides: New NLU overrides configuration (optional, structure as in __init__).
        """
        logger.info("Resetting TalkEngine with new configuration...")
        # Re-run the full initialization logic, including validation
        self._do_initialize(command_metadata, conversation_history, nlu_overrides)
        # Reset trained status
        self._is_trained = False
        logger.info("TalkEngine reset complete.")
