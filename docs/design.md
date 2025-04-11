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
The architecture centers around the `TalkEngine` class (`talkengine/engine.py`). This class encapsulates the NLU pipeline logic, holds the configuration (`command_metadata`, `conversation_history`, `nlu_overrides`), manages instances of the active NLU components (intent detection, parameter extraction, response generation), and orchestrates command execution based on the `nlu_overrides` configuration.

```mermaid
graph LR
    subgraph Configuration
        Meta[(Command Metadata<br/>desc, param_class)]
        Hist[(History)]
        Overrides[(NLU Overrides<br/>nlu_comps?, exec_code?)]
    end

    subgraph Engine
        TE(TalkEngine)
        subgraph NLU Components
            direction LR
            ID(IntentDetector)
            PE(ParamExtractor)
            TG(TextGenerator)
        end
        subgraph Default NLU
            direction LR
            DefID(DefaultIntentDetection)
            DefPE(DefaultParamExtraction)
            DefTG(DefaultResponseGeneration)
        end
    end

    subgraph Runtime
        UserApp -->|1. init(meta, hist?, overrides?)| TE
        TE -->|2. store| Meta
        TE -->|3. store| Hist
        TE -->|4. store| Overrides
        TE -->|5. _initialize_components| DefID
        TE -->| | DefPE
        TE -->| | DefTG
        Overrides -- NLU Overrides -->|6. select override OR default| ID
        Overrides -- NLU Overrides -->| | PE
        Overrides -- NLU Overrides -->| | TG
        TE -->|7. hold instance| ID
        TE -->| | PE
        TE -->| | TG

        UserApp -->|8. train()| TE
        TE -->|9. (placeholder)| TrainLogic

        UserApp -->|10. run(query, excluded_intents?)| TE
        Meta -->| | TE
        Hist -->| | TE
        TE -->|11. classify_intent(query, meta, hist, excluded_intents)| ID
        ID -->|12. intent_result| TE
        Meta -- param_class -->| | PE
        TE -->|13. identify_parameters(query, intent, param_class)| PE
        PE -->|14. parameters_dict| TE
        Overrides -- exec_code -->|15. lookup exec_code| TE
        subgraph Optional Code Execution [if exec_code defined in Overrides]
            TE -- parameters_dict -->|16. instantiate param_class| ParamObj(Parameter Object)
            TE -- exec_code.function -->|17. call function| ExecFunc{Executable Func}
            ParamObj -->| | ExecFunc
            ExecFunc -->|18. result_obj (instance of exec_code.result_class)| TE
        end
        TE -->|19. generate_text(intent, params_dict, result_obj?)| TG
        TG -->|20. response_text?| TE
        TE -->|21. NLUResult<br/>(command, params_dict, artifacts=result_obj?, ...)| UserApp

        UserApp -->|22. reset(meta, ...)| TE
        TE -->|23. re-init| DefID
        TE -->| | DefPE
        TE -->| | DefTG
    end

    style TE fill:#ccf,stroke:#333,stroke-width:2px
    style ID fill:#f9f,stroke:#333,stroke-width:1px
    style PE fill:#cdf,stroke:#333,stroke-width:1px
    style TG fill:#cfc,stroke:#333,stroke-width:1px
    style ExecFunc fill:#ffcc99,stroke:#333,stroke-width:1px
    style ParamObj fill:#eee,stroke:#666,stroke-width:1px
    style DefID fill:#f9f,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
    style DefPE fill:#cdf,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
    style DefTG fill:#cfc,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
```
The diagram illustrates that `TalkEngine` is initialized with `command_metadata` (containing description and `parameter_class`), `conversation_history`, and `nlu_overrides`. `nlu_overrides` can supply custom NLU components (ID, PE, TG) and/or command-specific `executable_code` (function and `result_class`). During `run`, the engine uses the active NLU components. Parameter extraction is guided by the `parameter_class` from the metadata. If `executable_code` is found in overrides for the identified command, the engine instantiates the `parameter_class` using extracted parameters, calls the specified function, and expects an instance of the specified `result_class` back. This result object (`artifacts`) and other NLU details are passed to the text generator. The final `NLUResult` (containing the command, parameter dict, and potentially the result object as `artifacts`) is returned.

### 2.2 Key Design Patterns
- **Strategy Pattern**: NLU logic (intent detection, parameter extraction, response generation) is defined by interfaces, allowing different implementations (strategies) to be used (defaults or developer-provided overrides via `nlu_overrides`).
- **Dependency Injection**: `TalkEngine` receives its core configuration (`command_metadata`, `conversation_history`, `nlu_overrides`) via its constructor.
- **Facade Pattern**: `TalkEngine` acts as a simplified interface (facade) over the underlying NLU pipeline steps, including optional code execution.
- **Command Pattern (Variant)**: The optional `executable_code` specification in `nlu_overrides` acts similarly to a command pattern, encapsulating a request (the function call) with its receiver logic and necessary context (parameter/result classes), although invoked directly by the engine rather than a separate invoker object.
- **Finite State Machine (FSM)**: The interaction handling mechanism (`InteractionState`, `InteractionHandler`) implements an FSM to manage sub-dialogues for clarification, validation, etc. The steps within these FSMs are tracked.

## 3. Component Design

### 3.1 `TalkEngine` Class (`talkengine/engine.py`)
- **Responsibilities**: Orchestrates the NLU process for a single query attempt. Holds configuration (`command_metadata`, `conversation_history`, `nlu_overrides`). Manages instances of active NLU components. Handles optional command execution by looking up `executable_code` in `nlu_overrides`, instantiating the command's `parameter_class` with extracted parameters, invoking the specified function, and validating the returned `result_class` instance. Manages transitions into and out of interaction modes. Provides `__init__`, `train`, `run`, and `reset` methods.
- **State**: Stores `_command_metadata` (dict[str, CommandDefinition]), `_conversation_history` (list[dict]), `_nlu_overrides` (dict), instances of the active NLU components (`_intent_detector`, `_param_extractor`, `_text_generator`), training status (`_is_trained`), and the current NLU pipeline context (`_pipeline_context` holding state like `current_intent`, `current_parameters` (dict), `artifacts` (Optional[BaseModel]), `interaction_mode`, etc.).
- **Methods**:
    - `__init__`: Stores provided configuration. Validates the structure of `command_metadata` (presence of `description`, `parameter_class` being a `BaseModel` subclass) and `nlu_overrides` (checking `executable_code` structure if present, including `function` being callable and `result_class` being a `BaseModel` subclass). Initializes the `NLUPipelineContext`, and calls `_initialize_components`.
    - `_initialize_components`: Instantiates default NLU components or uses overrides provided in `_nlu_overrides` for NLU interfaces. Passes necessary configuration (like `_command_metadata`) to constructors where needed (e.g., `DefaultParameterExtraction`).
    - `train`: Placeholder method.
    - `run(user_query: str, excluded_intents: Optional[list[str]] = None) -> NLUResult`: Takes a user query and optional excluded intents. Returns a single `NLUResult`. If in an interaction mode, delegates to the `InteractionHandler`. Otherwise, executes the main NLU pipeline:
        1. Intent classification using `_intent_detector` (passes `_command_metadata`, `_conversation_history`, `excluded_intents`). Updates context (`current_intent`).
        2. (Potentially enters/handles interaction mode for intent).
        3. Parameter extraction using `_param_extractor`. Requires the `parameter_class` for the current intent (retrieved from `_command_metadata`). Updates context (`current_parameters` dict).
        4. (Potentially enters/handles interaction mode for parameters).
        5. **Code Execution**: Check `_nlu_overrides` for an `executable_code` entry associated with the `context.current_intent`.
           - If found:
             a. Retrieve the `parameter_class` from `_command_metadata`.
             b. Retrieve `function` and `result_class` from the `executable_code` override.
             c. Instantiate the `parameter_class` using `context.current_parameters` dict (handle potential Pydantic validation errors).
             d. Call the `function`, passing the instantiated parameter object.
             e. Validate the returned object is an instance of `result_class` (handle potential type errors).
             f. Store the validated result object in `context.artifacts`.
           - If not found, `context.artifacts` remains `None`.
        6. **Text Generation**: Check if a text generator (`self._text_generator`) is configured. If yes, call `self._text_generator.generate_text(...)`, passing context including `command`, `parameters` (dict), and `artifacts` (`Optional[BaseModel]`). Store result in `text_response`.
        7. Construct `ConversationDetail` using `self._pipeline_context.recorded_interactions` and `text_response`.
        8. Construct final `NLUResult` using context values (`command`, `parameters` dict, `artifacts` object) and `ConversationDetail`.
        9. Return the `NLUResult`.
    - `reset`: Calls `__init__` to re-initialize with new configuration, performing the same validations.

### 3.2 NLU Interfaces (`talkengine/nlu_pipeline/nlu_engine_interfaces.py`)
- **`IntentDetectionInterface`**: Defines `classify_intent(query: str, command_metadata: dict[str, Any], conversation_history: list[dict], excluded_intents: Optional[list[str]]) -> Optional[str]`. (Note: `command_metadata` type hint might be refined later).
- **`ParameterExtractionInterface`**: Defines `identify_parameters(query: str, intent: str, parameter_class: Type[BaseModel], context: NLUPipelineContext) -> dict[str, Any]`. Takes the specific `parameter_class` for the identified intent.
- **`TextGenerationInterface`**: Defines `generate_text(command: str, parameters: dict[str, Any], artifacts: Optional[BaseModel], context: NLUPipelineContext) -> Optional[str]`. `artifacts` type hint updated to `Optional[BaseModel]`.

### 3.3 Interaction Handling Components (`talkengine/nlu_pipeline/interaction_*.py`)
- **`InteractionState` (Enum - `models.py`)**: Defines the possible states for interaction modes (e.g., `CLARIFYING_INTENT`, `VALIDATING_PARAMETER`).
- **`BaseInteractionData` & Subclasses (Models - `interaction_models.py`)**: Pydantic models defining the data payload required for each interaction mode (e.g., `ClarificationData` needs `options`).
- **`InteractionHandler` (ABC - `interaction_handlers.py`)**: Interface defining `get_initial_prompt` and `handle_input` methods. `handle_input` should return an `InteractionResult`. The handling logic should also trigger recording of interaction steps (prompt, response/feedback) into the `NLUPipelineContext`.
- **`InteractionResult` (Dataclass - `interaction_handlers.py`)**: Structured object returned by handlers indicating the next action (`response`, `exit_mode`, `proceed_immediately`, `update_context`).
- **Concrete Handlers (`ClarificationHandler`, etc. - `interaction_handlers.py`)**: Implementations of `InteractionHandler`. Their `handle_input` methods are responsible for processing user input *and* ensuring the interaction (prompt shown, user input received) is recorded in the `NLUPipelineContext`.

### 3.4 Default NLU Implementations (`talkengine/nlu_pipeline/default_*.py`)
- **`DefaultIntentDetection`**: Implements `IntentDetectionInterface`. Receives `command_metadata` during initialization or via the `classify_intent` call and utilizes the `description` field for classification.
- **`DefaultParameterExtraction`**: Implements `ParameterExtractionInterface`. Receives the target `parameter_class` in the `identify_parameters` call and uses its field definitions (names, types) to guide the extraction process from the query, returning a dictionary of extracted values.
- **`DefaultResponseGeneration`**: Implements `TextGenerationInterface`. `generate_text` handles `artifacts` being `Optional[BaseModel]` (e.g., using `str(artifacts)` or accessing specific known fields if possible) and returns a simple string representation including intent, parameters, and artifacts summary.

## 4. Data Models

- **Internal Command Definition (Conceptual/Helper Class)**:
    - Consider an internal `CommandDefinition` class/dataclass used within `TalkEngine` to hold the validated `description` and `parameter_class` loaded from `command_metadata`.
- **Configuration (Input)**:
    - `command_metadata` (dict[str, dict[str, Any]]): Developer-provided. Maps command keys to dictionaries containing:
        - `description`: str (Mandatory)
        - `parameter_class`: Type[BaseModel] (Mandatory, must be a Pydantic `BaseModel` subclass)
    - `conversation_history` (list[dict[str, Any]]): Optional. Developer-provided list of turn dictionaries.
    - `nlu_overrides` (dict[str, Any]): Optional. Can contain keys for NLU interfaces (e.g., `"intent_detection"`) mapping to implementation instances, AND/OR command name keys mapping to dictionaries. A command name dictionary currently supports one key: `"executable_code"`, which maps to another dictionary containing:
        - `function`: Callable (Mandatory)
        - `result_class`: Type[BaseModel] (Mandatory, must be a Pydantic `BaseModel` subclass)
- **NLU Pipeline Context (`models.py` - `NLUPipelineContext`)**:
    - `command_metadata`: Store the validated input `command_metadata`. Type hint `dict[str, dict[str, Any]]` or potentially the internal `CommandDefinition` type.
    - `current_parameters`: Remains `dict[str, Any]`. Stores the output from `ParameterExtractionInterface`.
    - `artifacts`: Updated type hint to `Optional[BaseModel]`. Stores the result object from `executable_code` function.
- **Interaction Data (`interaction_models.py`)**: No changes.
- **Interaction Log Entry:** No changes.
- **NLU Result (Output - `NLUResult` in `models.py`)**:
    - `command`: `Optional[str]`
    - `parameters`: `Optional[dict[str, Any]]`. Describes the values extracted, corresponding to fields in the command's `parameter_class`.
    - `artifacts`: `Optional[BaseModel]`. Stores the validated instance returned by the `executable_code` function.
    - `conversation_detail`: `ConversationDetail`
- **Conversation Detail (Output - `ConversationDetail` in `models.py`)**: No changes.

## 5. Process Flows

### 5.1 Initialization (`TalkEngine(...)`)
1. Developer instantiates `TalkEngine` (`command_metadata`, `history`?, `overrides`?).
2. `__init__` stores args.
3. `__init__` validates `command_metadata`: Checks each command has `description` (str) and `parameter_class` (is subclass of `BaseModel`). Stores internally (maybe as `dict[str, CommandDefinition]`).
4. `__init__` validates `nlu_overrides`: Checks structure of NLU component overrides. Checks structure of command `executable_code` overrides (`function` is callable, `result_class` is subclass of `BaseModel`).
5. `__init__` initializes `NLUPipelineContext`.
6. `__init__` calls `_initialize_components` to set up active NLU instances based on defaults and `nlu_overrides`.

### 5.2 Query Processing (`engine.run(query, excluded_intents?) -> NLUResult`)
1. Developer calls `engine.run(user_query, excluded_intents)`. Reset context fields for this run (e.g., `artifacts`, `parameters`).
2. `run` checks `self._pipeline_context.interaction_mode`.
3. **If in Interaction Mode:** Delegate to handler, process result, potentially return intermediate `NLUResult`. (No change in core logic here).
4. **If NOT in Interaction Mode:**
    a. Call `self._intent_detector.classify_intent(...)`. Update `context.current_intent`.
    b. (Handle Intent Interaction if needed).
    c. If intent found: Retrieve the `parameter_class` for the intent from `_command_metadata`.
    d. Call `self._param_extractor.identify_parameters(query, intent, parameter_class, context)`. Update `context.current_parameters` (dict).
    e. (Handle Parameter Interaction if needed).
    f. Check `_nlu_overrides` for `executable_code` for the `context.current_intent`.
        i. If yes: Retrieve `function`, `result_class`, and `parameter_class`.
        ii. Try instantiating `parameter_class(**context.current_parameters)`. Catch Pydantic `ValidationError`.
        iii. Try calling `function(parameter_object)`. Catch general exceptions.
        iv. Check if result is `instanceof(result_class)`. Raise `TypeError` if not.
        v. Store validated result in `context.artifacts`.
        vi. If any error occurs in steps ii-iv, handle appropriately (e.g., log error, potentially set an error state in `NLUResult`, skip text generation or provide error text).
    g. Check if `self._text_generator` is configured.
        i. If yes: Call `self._text_generator.generate_text(command, parameters, artifacts, context)`. Store in `text_response`.
    h. Construct `ConversationDetail` (using recorded interactions, `text_response`).
    i. Construct final `NLUResult` (using `context.command`, `context.parameters`, `context.artifacts`, `ConversationDetail`).
    j. Return `NLUResult`.

### 5.3 Reset (`engine.reset(...)`)
1. Developer calls `engine.reset()` with new configuration.
2. `reset` calls `self.__init__()` with the new arguments.
3. The initialization flow (Section 5.1) repeats, including validations.

## 6. Implementation Considerations

### 6.1 Technology Stack
- **Core Language**: Python 3.12+
- **Data Validation/Modeling**: Pydantic (Core dependency)
- **NLU Interfaces**: `abc` module.
- **Logging**: Standard Python `logging`.

### 6.2 Error Handling
- Add explicit validation for `command_metadata` and `nlu_overrides` structures in `__init__` and `reset`.
- Handle potential Pydantic `ValidationError` during `parameter_class` instantiation within `run`.
- Handle potential `TypeError` if the `executable_code` function returns an object of the wrong type.
- Handle general exceptions during `executable_code` function execution.
- Add `try...except` blocks around calls to custom NLU component methods.

### 6.3 Extensibility
- Primary extension mechanisms remain providing custom NLU component implementations and command `executable_code` via `nlu_overrides`.

## 7. Testing Strategy
- Focus on testing the `TalkEngine` class behavior.
- Use `unittest.mock` for NLU component interfaces.
- Test `TalkEngine` initialization/reset with valid and invalid `command_metadata` and `nlu_overrides` structures (checking for expected validation errors).
- Test `TalkEngine.run`:
    - Correct retrieval and use of `parameter_class` by `ParameterExtractionInterface`.
    - Correct lookup and execution flow for `executable_code` from `nlu_overrides`.
    - Correct instantiation of `parameter_class` from extracted parameters dict (test valid and invalid cases).
    - Correct validation of the `result_class` instance returned by the executable function (test correct type, incorrect type).
    - Handling of exceptions during parameter instantiation and function execution.
    - Correct population of `NLUResult` fields (`parameters` dict, `artifacts` BaseModel instance).
    - Correct passing of `artifacts` (BaseModel instance) to `TextGenerationInterface`.
- Update tests for `DefaultParameterExtraction` to use `parameter_class`.
- Update tests for `DefaultResponseGeneration` to handle `artifacts` as `Optional[BaseModel]`.
- Add tests for interaction handling remains relevant.
- Tests for `excluded_intents` remain relevant. 