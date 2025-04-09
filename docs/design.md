# talkengine - Architecture & Design (Streamlined)

## 1. Introduction

### 1.1 Purpose
This document describes the architectural design and implementation details of the streamlined `talkengine` library, focusing on the `TalkEngine` class.

### 1.2 Scope
This design document covers the `TalkEngine` class, its core components (NLU interfaces and default implementations), data flow, and interaction patterns.

### 1.3 Design Goals
- **Simplicity**: Provide a single, easy-to-use class (`TalkEngine`) as the main entry point.
- **Configurability**: Allow developers to inject command metadata, history, and custom NLU logic easily during initialization.
- **Modularity**: Maintain separation between the engine orchestration (`TalkEngine`) and the NLU component interfaces/implementations.

## 2. Architectural Overview

### 2.1 High-Level Architecture
The architecture centers around the `TalkEngine` class (`talkengine/engine.py`). This class encapsulates the NLU pipeline logic, holds instances of NLU components (intent detection, parameter extraction, text generation), and directly handles an optional code execution step based on command metadata.

```mermaid
graph LR
    UserApp -->|1. init(meta, hist?, overrides?)| TE(TalkEngine)
    TE -->|2. store| Meta[(Metadata)]
    TE -->|3. store| Hist[(History)]
    TE -->|4. store| Overrides[(NLU Overrides)]
    TE -->|5. _initialize_components| DefID(DefaultIntentDetection)
    TE -->| | DefPE(DefaultParamExtraction)
    TE -->| | DefTG(DefaultResponseGeneration)
    Overrides -->|6. select override OR default| ID(IntentDetector)
    Overrides -->| | PE(ParamExtractor)
    Overrides -->| | TG(TextGenerator)
    TE -->|7. hold instance| ID
    TE -->| | PE
    TE -->| | TG

    UserApp -->|8. train()| TE
    TE -->|9. (placeholder)| TrainLogic

    UserApp -->|10. run(query, excluded_intents?)| TE
    TE -->|11. classify_intent(query, excluded_intents)| ID
    ID -->|12. intent_result| TE
    TE -->|13. identify_parameters(query, intent)| PE
    PE -->|14. parameters| TE
    TE -->|15. execute_code?(intent, params)| InternalLogic
    InternalLogic -->|16. code_result?| TE
    TE -->|17. generate_text(intent, params, code_result?)| TG
    TG -->|18. response_text?| TE
    TE -->|19. NLUResult| UserApp

    UserApp -->|20. reset(meta, ...)| TE
    TE -->|21. re-init| DefID
    TE -->| | DefPE
    TE -->| | DefTG

    style TE fill:#ccf,stroke:#333,stroke-width:2px
    style ID fill:#f9f,stroke:#333,stroke-width:1px
    style PE fill:#cdf,stroke:#333,stroke-width:1px
    style InternalLogic fill:#ffcc99,stroke:#333,stroke-width:1px,label:"Internal Code Exec"
    style TG fill:#cfc,stroke:#333,stroke-width:1px
    style DefID fill:#f9f,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
    style DefPE fill:#cdf,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
    style DefTG fill:#cfc,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
```

### 2.2 Key Design Patterns
- **Strategy Pattern**: NLU logic (intent detection, parameter extraction, text generation) is defined by interfaces, allowing different implementations (strategies) to be used (defaults or developer-provided overrides).
- **Dependency Injection**: `TalkEngine` receives its core configuration (metadata, history, overrides) via its constructor.
- **Facade Pattern**: `TalkEngine` acts as a simplified interface (facade) over the underlying NLU pipeline steps.
- **Finite State Machine (FSM)**: The interaction handling mechanism (`InteractionState`, `InteractionHandler`) implements an FSM to manage sub-dialogues for clarification, validation, etc. The steps within these FSMs are tracked.

## 3. Component Design

### 3.1 `TalkEngine` Class (`talkengine/engine.py`)
- **Responsibilities**: Orchestrates the NLU process for a single query attempt, potentially managing transitions into and out of interaction modes. Holds configuration (metadata, history, NLU overrides). Manages instances of NLU components (intent detector, param extractor, text generator). Directly handles optional code execution based on metadata. Provides `__init__`, `train`, `run`, and `reset` methods.
- **State**: Stores command metadata, conversation history, NLU overrides, instances of the active NLU components (`_intent_detector`, `_param_extractor`, `_text_generator`), training status (`_is_trained`), and the current NLU pipeline context (`_pipeline_context` holding state like `current_intent`, `current_parameters`, `interaction_mode`, `interaction_data`, `recorded_interactions`).
- **Methods**:
    - `__init__`: Stores provided configuration, initializes the `NLUPipelineContext`, and calls `_initialize_components`.
    - `_initialize_components`: Instantiates default NLU components (`DefaultIntentDetection`, etc.) or uses overrides provided in `nlu_overrides`. Passes necessary configuration (like `command_metadata`) to constructors.
    - `train`: Placeholder method.
    - `run(user_query: str, excluded_intents: Optional[list[str]] = None) -> NLUResult`: Takes a user query and optional list of intents to exclude. If in an interaction mode, delegates input to the appropriate `InteractionHandler`. Otherwise, executes the main NLU pipeline:
        1. Intent classification using `_intent_detector` (respecting `excluded_intents`).
        2. Parameter extraction using `_param_extractor`.
        3. (Potentially enters/handles interaction mode).
        4. Looks up `executable_code` in `_command_metadata` for the identified intent. If found, executes it with the extracted parameters, storing the result.
        5. Optional text generation using `_text_generator`.
        6. Updates `_pipeline_context`.
        7. Constructs and returns an `NLUResult` object.
    - `reset`: Calls `__init__` to re-initialize with new configuration.

### 3.2 NLU Interfaces (`talkengine/nlu_pipeline/nlu_engine_interfaces.py`)
- **`IntentDetectionInterface`**: Defines `classify_intent(...)`.
- **`ParameterExtractionInterface`**: Defines `identify_parameters(...)`.
- **`TextGenerationInterface`**: Defines `generate_text(intent, parameters, code_execution_result, pipeline_context)` method signature, returning the user-facing response text (str). `code_execution_result` will be `None` if no code was executed.

### 3.3 Interaction Handling Components (`talkengine/nlu_pipeline/interaction_*.py`)
- **`InteractionState` (Enum - `models.py`)**: Defines the possible states for interaction modes (e.g., `CLARIFYING_INTENT`, `VALIDATING_PARAMETER`).
- **`BaseInteractionData` & Subclasses (Models - `interaction_models.py`)**: Pydantic models defining the data payload required for each interaction mode (e.g., `ClarificationData` needs `options`).
- **`InteractionHandler` (ABC - `interaction_handlers.py`)**: Interface defining `get_initial_prompt` and `handle_input` methods. `handle_input` should return an `InteractionResult`. The handling logic should also trigger recording of interaction steps (prompt, response/feedback) into the `NLUPipelineContext`.
- **`InteractionResult` (Dataclass - `interaction_handlers.py`)**: Structured object returned by handlers indicating the next action (`response`, `exit_mode`, `proceed_immediately`, `update_context`).
- **Concrete Handlers (`ClarificationHandler`, etc. - `interaction_handlers.py`)**: Implementations of `InteractionHandler`. Their `handle_input` methods are responsible for processing user input *and* ensuring the interaction (prompt shown, user input received) is recorded in the `NLUPipelineContext`.

### 3.4 Default NLU Implementations (`talkengine/nlu_pipeline/default_*.py`)
- **`DefaultIntentDetection`**: Implements `IntentDetectionInterface`.
- **`DefaultParameterExtraction`**: Implements `ParameterExtractionInterface`.
- **`DefaultResponseGeneration`**: Implements `TextGenerationInterface`. `generate_text` returns a simple string representation including intent, parameters, and optionally the `code_execution_result` (passed in by `TalkEngine`).

## 4. Data Models

- **Configuration (Input)**:
    - `command_metadata` (dict): Developer-provided. Maps command keys to dictionaries containing `description` (str), optionally `parameters` (dict), and optionally `executable_code` (e.g., a function reference).
    - `conversation_history` (list[dict]): Optional. Developer-provided list of turn dictionaries (e.g., `{"role": "user", "content": "..."}`). Structure is flexible, consumed by NLU implementations.
    - `nlu_overrides` (dict): Optional. Maps NLU interface names (`"intent_detection"`, `"param_extraction"`, `"text_generation"`) to instances implementing the respective interfaces.
- **NLU Pipeline Context (`models.py`)**:
    - `NLUPipelineContext`: Internal state object holding `current_state`, `current_intent`, `current_parameters`, `interaction_mode`, `interaction_data`, `recorded_interactions` (list to store interaction tuples/objects), etc.
- **Interaction Data (`interaction_models.py`)**:
    - `BaseInteractionData` subclasses (e.g., `ClarificationData`, `ValidationData`): Data passed into handlers via `NLUPipelineContext.interaction_data`.
- **Interaction Log Entry (Conceptual - stored in `recorded_interactions`)**:
    - Structure TBD, e.g., `Tuple[str, str, str]` representing `(interaction_stage_type, prompt_shown, user_response)`.
- **NLU Result (Output - Pydantic BaseModel in `models.py`)**:
    - `NLUResult`: Returned by `TalkEngine.run()`. Contains:
        - `command: str`
        - `parameters: Optional[dict]`
        - `confidence: Optional[float]`
        - `code_execution_result: Optional[dict]`
        - `conversation_detail: ConversationDetail`
- **Conversation Detail (Output - Pydantic BaseModel in `models.py`)**:
    - `ConversationDetail`: Nested within `NLUResult`. Contains:
        - `interactions: list` # List of interaction log entries
        - `response_text: Optional[str]`

## 5. Process Flows

### 5.1 Initialization (`TalkEngine(...)`)
1. Developer instantiates `TalkEngine` (metadata, history?, overrides?).
2. `__init__` stores args.
3. `__init__` calls `_initialize_components`.
4. `_initialize_components` checks `_nlu_overrides` for each NLU interface type (`intent_detection`, `param_extraction`, `text_generation`).
5. If override exists, store it (e.g., `self._intent_detector`).
6. If no override, instantiate default (e.g., `DefaultIntentDetection`) and store it.

### 5.2 Query Processing (`engine.run(query, excluded_intents?) -> NLUResult`)
1. Developer calls `engine.run(user_query, excluded_intents)`. Context's `recorded_interactions` is cleared/initialized for this run.
2. `run` checks `self._pipeline_context.interaction_mode`.
3. **If in an Interaction Mode:**
    a. Retrieve `InteractionHandler`.
    b. Call `handler.handle_input(user_query, self._pipeline_context)`. This method internally records the interaction step (e.g., appends `('clarification', prompt, user_query)` to `_pipeline_context.recorded_interactions`).
    c. Process `InteractionResult`: Update context, potentially exit mode. If exiting and proceeding, continue to step 4. Otherwise, construct and return `NLUResult` using current context state (command might be `None` or partially resolved, `response_text` might be the interaction prompt).
4. **If NOT in an Interaction Mode:**
    a. Call `self._intent_detector.classify_intent(user_query, self._pipeline_context, excluded_intents)`. Update context (`current_intent`, `current_confidence`).
    b. **Decision Point (Intent):** If clarification needed, set `interaction_mode`, create data, get `InteractionHandler`, call `get_initial_prompt`, record interaction step (`('clarification_start', prompt, None)`), return `NLUResult` with current state and interaction prompt as `response_text`.
    c. Call `self._param_extractor.identify_parameters(user_query, context.current_intent, self._pipeline_context)`. Update context (`current_parameters`).
    d. **Decision Point (Params):** If validation needed, handle interaction...
    e. Check `_command_metadata` for `executable_code` associated with `context.current_intent`.
       If found, execute the code with `context.current_parameters`, store the result in `code_exec_result`.
       If not found or not specified, set `code_exec_result = None`.
    f. Check if a text generator (`self._text_generator`) is configured. If yes, call `self._text_generator.generate_text(...)` with `code_exec_result` (which might be `None`). Store result in `text_response`.
      If no text generator is configured, set `text_response = None`.
    g. Construct `ConversationDetail` using `self._pipeline_context.recorded_interactions` and `text_response` (which might be `None`).
    h. Construct final `NLUResult` using `context.current_intent`, `context.current_parameters`, `context.current_confidence`, `code_exec_result` (which might be `None`), and the constructed `ConversationDetail`.
    i. Return the `NLUResult`.

### 5.3 Reset (`engine.reset(...)`)
1. Developer calls `engine.reset()` with new configuration.
2. `reset` calls `self.__init__()` with the new arguments.
3. The initialization flow (Steps 1-7 in 5.1) repeats, overwriting previous state and components.

## 6. Implementation Considerations

### 6.1 Technology Stack
- **Core Language**: Python 3.12+
- **NLU Interfaces**: `abc` module.
- **Logging**: Standard Python `logging`.

### 6.2 Error Handling
- Current implementation relies on Python's standard error handling.
- Potential Enhancement: Add explicit validation for input `command_metadata` structure in `__init__`.
- Potential Enhancement: Add `try...except` blocks within `TalkEngine.run` around calls to NLU component methods to handle potential errors raised by custom implementations.

### 6.3 Extensibility
- Primary extension mechanism is providing custom NLU component implementations via the `nlu_overrides` dictionary during `TalkEngine` initialization.
- Interaction handlers (`ClarificationHandler`, etc.) provide points for customizing dialogue behavior.

## 7. Testing Strategy
- Focus on testing the `TalkEngine` class behavior.
- Use `unittest.mock` to mock NLU component interfaces (`@pytest.fixture`) to test `TalkEngine.run`'s orchestration logic and output structure independently of default implementations.
- Test `TalkEngine` initialization with defaults and overrides.
- Test `TalkEngine.reset` ensures state is correctly updated.
- Add separate tests for the logic within default NLU implementations once finalized.
- Add tests for each `InteractionHandler` to verify prompt generation and input processing logic.
- Add tests for `TalkEngine.run` covering transitions into and out of different interaction modes.
- Test `TalkEngine.run` with `excluded_intents` to ensure intents are correctly filtered.
- Add tests ensuring `TalkEngine.run` correctly looks up and executes `executable_code` from metadata and passes the result to the text generator.
- Test validation in `_initialize_components` ensuring commands have either `executable_code` in metadata or a text generator configured. 