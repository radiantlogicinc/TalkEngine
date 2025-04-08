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
The architecture centers around the `TalkEngine` class (`talkengine/engine.py`). This class encapsulates the NLU pipeline logic and holds instances of NLU components that adhere to defined interfaces (`talkengine/nlu_pipeline/nlu_engine_interfaces.py`). Default implementations for these interfaces are provided (`talkengine/nlu_pipeline/default_*.py`).

```mermaid
graph LR
    UserApp -->|1. init(meta, hist?, overrides?)| TE(TalkEngine)
    TE -->|2. store| Meta[(Metadata)]
    TE -->|3. store| Hist[(History)]
    TE -->|4. store| Overrides[(Overrides)]
    TE -->|5. _initialize_nlu_components| DefID(DefaultIntentDetection)
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

    UserApp -->|10. run(query)| TE
    TE -->|11. classify_intent(query)| ID
    ID -->|12. intent_result| TE
    TE -->|13. identify_parameters(query, intent)| PE
    PE -->|14. parameters| TE
    TE -->|15. generate_text(intent, params)| TG
    TG -->|16. raw_response, text_response| TE
    TE -->|17. (nlu_result, hint)| UserApp

    UserApp -->|18. reset(meta, ...)| TE
    TE -->|19. re-init| DefID
    TE -->| | DefPE
    TE -->| | DefTG

    style TE fill:#ccf,stroke:#333,stroke-width:2px
    style ID fill:#f9f,stroke:#333,stroke-width:1px
    style PE fill:#cdf,stroke:#333,stroke-width:1px
    style TG fill:#cfc,stroke:#333,stroke-width:1px
    style DefID fill:#f9f,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
    style DefPE fill:#cdf,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
    style DefTG fill:#cfc,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
```

### 2.2 Key Design Patterns
- **Strategy Pattern**: NLU logic (intent detection, parameter extraction, text generation) is defined by interfaces, allowing different implementations (strategies) to be used (defaults or developer-provided overrides).
- **Dependency Injection**: `TalkEngine` receives its core configuration (metadata, history, overrides) via its constructor.
- **Facade Pattern**: `TalkEngine` acts as a simplified interface (facade) over the underlying NLU pipeline steps.

## 3. Component Design

### 3.1 `TalkEngine` Class (`talkengine/engine.py`)
- **Responsibilities**: Orchestrates the NLU process for a single query. Holds configuration (metadata, history, overrides). Manages instances of NLU components. Provides `__init__`, `train`, `run`, and `reset` methods.
- **State**: Stores command metadata, conversation history, NLU overrides, instances of the active NLU components (`_intent_detector`, `_param_extractor`, `_text_generator`), and training status (`_is_trained`).
- **Methods**: 
    - `__init__`: Stores provided configuration and calls `_initialize_nlu_components`.
    - `_initialize_nlu_components`: Instantiates default NLU components (`DefaultIntentDetection`, etc.) or uses overrides provided in `nlu_overrides`. Passes necessary configuration (like `command_metadata`) to the component constructors.
    - `train`: Placeholder method.
    - `run`: Takes a user query, calls `classify_intent` on the intent detector, `identify_parameters` on the parameter extractor, and `generate_text` on the text generator. Formats and returns the NLU result dictionary and hint string.
    - `reset`: Calls `__init__` to re-initialize with new configuration.

### 3.2 NLU Interfaces (`talkengine/nlu_pipeline/nlu_engine_interfaces.py`)
- **`IntentDetectionInterface`**: Defines the `classify_intent(user_input, excluded_intents=None)` method signature, which must return a dictionary containing at least `intent` (str) and `confidence` (float).
- **`ParameterExtractionInterface`**: Defines the `identify_parameters(user_input, intent)` method signature, which must return a dictionary of extracted parameters.
- **`TextGenerationInterface`**: Defines the `generate_text(intent, parameters)` method signature, which must return a tuple containing the raw response data (Any) and the user-facing response text (str).

### 3.3 Default NLU Implementations (`talkengine/nlu_pipeline/default_*.py`)
- **`DefaultIntentDetection`**: Implements `IntentDetectionInterface`. Takes `command_metadata` in `__init__`. Uses basic keyword/substring matching against command keys to implement `classify_intent`.
- **`DefaultParameterExtraction`**: Implements `ParameterExtractionInterface`. Takes `command_metadata` in `__init__`. `identify_parameters` currently returns an empty dictionary (placeholder).
- **`DefaultResponseGeneration`**: Implements `TextGenerationInterface`. `generate_text` returns a dictionary containing the intent and parameters as the raw response, and a simple string representation as the text response.

## 4. Data Models

- **Configuration (Input)**:
    - `command_metadata` (dict): Developer-provided. Maps command keys (str) to dictionaries containing at least a `description` (str) and optionally `parameters` (dict).
    - `conversation_history` (list[dict]): Optional. Developer-provided list of turn dictionaries (e.g., `{"role": "user", "content": "..."}`). Structure is flexible, consumed by NLU implementations.
    - `nlu_overrides` (dict): Optional. Developer-provided. Maps interface names (`"intent_detection"`, `"param_extraction"`, `"text_generation"`) to instances of classes implementing the respective interfaces.
- **NLU Result (Output)**:
    - `nlu_result` (dict): Returned by `TalkEngine.run()`. Contains keys: `intent` (str), `parameters` (dict), `confidence` (float), `raw_response` (Any), `response_text` (str).

## 5. Process Flows

### 5.1 Initialization (`TalkEngine(...)`)
1.  Developer instantiates `TalkEngine`, passing `command_metadata`, optional `conversation_history`, and optional `nlu_overrides`.
2.  `__init__` stores these arguments internally.
3.  `__init__` calls `_initialize_nlu_components`.
4.  `_initialize_nlu_components` checks `_nlu_overrides` for each interface type (`intent_detection`, etc.).
5.  If an override exists and is valid, its instance is stored (e.g., in `self._intent_detector`).
6.  If no valid override exists, an instance of the corresponding default implementation (e.g., `DefaultIntentDetection`) is created (passing `_command_metadata` if needed) and stored.

### 5.2 Query Processing (`engine.run(query)`)
1.  Developer calls `engine.run(user_query)`.
2.  `run` calls `self._intent_detector.classify_intent(user_query)`.
3.  `run` receives the intent result (dict with `intent`, `confidence`).
4.  `run` calls `self._param_extractor.identify_parameters(user_query, identified_intent)`.
5.  `run` receives the parameters dictionary.
6.  `run` calls `self._text_generator.generate_text(identified_intent, parameters)`.
7.  `run` receives the raw response and text response.
8.  `run` constructs the `nlu_result` dictionary.
9.  `run` sets `hint = "new_conversation"`.
10. `run` returns `(nlu_result, hint)`. 

### 5.3 Reset (`engine.reset(...)`)
1.  Developer calls `engine.reset()` with new configuration.
2.  `reset` calls `self.__init__()` with the new arguments.
3.  The initialization flow (Steps 1-6 in 5.1) repeats, overwriting previous state and components.

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

## 7. Testing Strategy
- Focus on testing the `TalkEngine` class behavior.
- Use `unittest.mock` to mock NLU component interfaces (`@pytest.fixture`) to test `TalkEngine.run`'s orchestration logic and output structure independently of default implementations.
- Test `TalkEngine` initialization with defaults and overrides.
- Test `TalkEngine.reset` ensures state is correctly updated.
- Add separate tests for the logic within default NLU implementations once finalized. 