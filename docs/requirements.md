# talkengine - Software Requirements Specification (Streamlined)

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for the streamlined `talkengine` library, focusing on providing a simple NLU pipeline via the `TalkEngine` class.

### 1.2 Product Scope
`talkengine` allows developers to initialize an NLU pipeline (`TalkEngine`) with command descriptions (metadata), optional conversation history, and optional NLU component overrides. The engine processes natural language queries, potentially engaging in interactive sub-dialogues (e.g., for clarification or validation) to identify user intent and extract parameters based on the provided configuration.

### 1.3 Intended Audience
This document is intended for developers using the `talkengine` library.

## 2. Overall Description

### 2.1 Product Perspective
`talkengine` is a lightweight Python library providing a configurable NLU pipeline object (`TalkEngine`) for basic intent classification and parameter extraction from single user queries.

### 2.2 Product Features
- **Configurable Pipeline Initialization**: `TalkEngine` is initialized directly with command metadata, optional history, and optional NLU overrides (as Python objects/functions).
- **Single Query Processing**: The `run()` method processes one user query at a time.
- **Intent Classification**: Maps user input to a predefined command key based on the provided metadata and configured intent detector.
- **Parameter Extraction**: Identifies potential parameters based on the user input, classified intent, and configured parameter extractor.
- **Interactive Sub-dialogues**: Can enter specific modes to:
    - Clarify ambiguous user intent.
    - Validate or request missing parameters.
    - Solicit feedback on previous responses.
- **Text Generation**: Generates a basic textual summary and optionally returns artifacts (a dictionary of result data from the optional code execution step).
- **Hint Generation**: Provides a hint indicating if the query is treated as a new conversation turn (currently always `'new_conversation'`).
- **Stateful Reset**: The `reset()` method allows re-initializing the `TalkEngine` instance with new configuration.

### 2.3 User Classes and Characteristics
Python developers needing a simple, configurable NLU pipeline for single-turn query processing.

### 2.4 Operating Environment
- Python 3.12+
- Dependencies: (Check `pyproject.toml` - currently primarily standard libraries + `pytest` for dev)

## 3. Functional Requirements

### 3.1 `TalkEngine` Initialization & Reset
FR-1.1: The system shall provide a `TalkEngine` class.
FR-1.2: `TalkEngine` shall be initializable with a command metadata dictionary.
FR-1.3: `TalkEngine` shall accept optional conversation history (list of dicts) during initialization.
FR-1.4: `TalkEngine` shall accept optional NLU component overrides (dictionary mapping interface names to implementation instances/functions) during initialization.
FR-1.5: `TalkEngine` shall provide a `reset()` method to re-initialize the instance with new metadata, history, and overrides.
FR-1.6: `TalkEngine` shall provide a placeholder `train()` method.

### 3.2 NLU Processing (`run` method)
FR-2.1: `TalkEngine` shall provide a `run(query: str)` method.
FR-2.2: The `run()` method shall process the input query using the configured NLU components (intent detector, parameter extractor) and internal logic for response generation (optional code execution, mandatory text generation).
FR-2.3: The `run()` method shall return a tuple containing:
    a) The original user query string.
    b) A list of `NLUResult` objects. Each object represents one attempt to process the query through the NLU pipeline. Initially, this list will contain one result. Subsequent results may be added if the calling application triggers a re-run based on feedback (See FR-2.9).
FR-2.4: Each `NLUResult` object shall be a Pydantic BaseModel containing:
    a) `command` (str): The identified command key from the metadata, or `'unknown'`.
    b) `parameters` (Optional[dict]): A dictionary of extracted parameter names and values.
    d) `artifacts` (Optional[dict]): The result returned from executing code associated with the identified command, if any.
    e) `conversation_detail` (ConversationDetail): An object containing details about the interaction flow for this attempt (See FR-2.5).
FR-2.5: The `ConversationDetail` object within `NLUResult` shall be a Pydantic BaseModel containing:
    a) `interactions` (list): A list of tuples or objects, each documenting a step in any interactive sub-dialogue (e.g., clarification, validation). Each entry should record the interaction stage/type, the prompt presented to the user, and the user's feedback/response during that interaction step.
    b) `response_text` (Optional[str]): The final user-facing text generated for this attempt, if any.
FR-2.6: Intent classification shall identify a command key defined in the metadata (or `'unknown'`), contributing to the `command` field of the `NLUResult`.
FR-2.7: Parameter extraction shall identify parameter names and values, contributing to the `parameters` field of the `NLUResult`.
FR-2.8: The response generation phase shall consist of two optional steps orchestrated by the `TalkEngine`:
    a) Code Execution: If executable code (e.g., a function reference) is associated with the identified command in the `command_metadata`, the `TalkEngine` shall execute it with the extracted `parameters`. The result can help populate the `artifacts` field of the `NLUResult`.
    b) Text Generation: If a text generator component is configured (via overrides or default), the `TalkEngine` shall invoke it to generate a user-facing response based on the `command`, `parameters`, and optional `artifacts`. The result populates the `response_text` field of the `ConversationDetail`.
    c) Code Execution is optional but text generation is not. A command may be configured with just text generation or both code execution and text generation.
FR-2.9: The `run` method shall accept an optional argument to exclude specific command intents from the classification step. This allows a calling application to re-invoke `run` if initial results are unsatisfactory (e.g., user feedback indicates the wrong command was inferred), preventing the same incorrect command from being chosen again in the subsequent attempt. The new `NLUResult` from the re-run is appended to the list returned by `run`.

### 3.3 Interaction Handling
FR-3.1: The system shall support entering distinct interaction modes when needed (e.g., intent clarification, parameter validation).
FR-3.2: The system shall provide mechanisms (handlers) to manage the dialogue flow within each interaction mode, including generating prompts and processing user responses specific to that mode.
FR-3.3: The system shall define specific data structures to carry the necessary context and information for each interaction mode (e.g., clarification options, parameter to validate).
FR-3.4: Each step taken within an interaction mode (prompt generated, user feedback received) shall be recorded in the `interactions` list within the `ConversationDetail` object of the current `NLUResult`.
FR-3.5: Upon successful completion of an interaction mode, the system shall update its internal state (e.g., clarified intent, validated parameter) and potentially resume the main NLU pipeline processing for the current attempt.
FR-3.6: The system shall allow exiting an interaction mode based on user input or internal logic.

## 4. Non-Functional Requirements

### 4.1 Usability
NFR-1.1: The `TalkEngine` API shall be simple and intuitive to use.
NFR-1.2: Configuration via dictionaries and optional overrides should be straightforward.
NFR-1.3: Prompts generated during interaction modes shall be clear and guide the user effectively.

### 4.2 Reliability
NFR-2.1: The system should handle basic errors during initialization (e.g., missing metadata) gracefully (Future enhancement: Add more robust validation).
NFR-2.2: Default NLU implementations should provide baseline functionality.

### 4.3 Supportability
NFR-3.1: The library shall include clear docstrings for the `TalkEngine` class and its methods.
NFR-3.2: A README file shall provide a quick start example.

## 5. Quick Start Example (from README.md)

```python
import logging
from talkengine import TalkEngine

# Configure logging to see internal steps
logging.basicConfig(level=logging.INFO)

# 1. Define your command metadata
command_metadata = {
    "calculator.add": {
        "description": "Adds two numbers together.",
        "parameters": {"num1": "int", "num2": "int"} # Parameter types are informational for now
    },
    "weather.get_forecast": {
        "description": "Gets the weather forecast for a location.",
        "parameters": {"location": "str", "date": "str"}
    },
}

# 2. (Optional) Define conversation history
conversation_history = [
    {"role": "user", "content": "What was the weather yesterday?"},
    # Add previous NLU results if available for context
]

# 3. (Optional) Define NLU overrides (using default implementations here)
# See example in talkengine/engine.py for custom override structure
nlu_overrides = {}

# 4. Initialize the engine
engine = TalkEngine(
    command_metadata=command_metadata,
    conversation_history=conversation_history, # Optional
    nlu_overrides=nlu_overrides # Optional
)

# 5. Train the engine (currently a placeholder)
engine.train()

# 6. Process queries in a loop
print("\nTalkEngine Ready. Type 'quit' to exit.")
while True:
    user_query = input("> ")
    if user_query.lower() == 'quit':
        break

    nlu_result, hint = engine.run(user_query)

    print(f"  Hint: {hint}") # Currently always "new_conversation"
    print(f"  Intent: {nlu_result.get('intent', 'N/A')}")
    print(f"  Parameters: {nlu_result.get('parameters', {})}")
    # print(f"  Raw Response: {nlu_result.get('raw_response')}") # Internal data structure
    print(f"  Response Text: {nlu_result.get('response_text')}") # Basic text summary
``` 