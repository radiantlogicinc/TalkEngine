# talkengine - Software Requirements Specification (Streamlined)

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for the streamlined `talkengine` library, focusing on providing a simple NLU pipeline via the `TalkEngine` class.

### 1.2 Product Scope
`talkengine` allows developers to initialize an NLU pipeline (`TalkEngine`) with command descriptions (metadata), optional conversation history, and optional NLU component overrides. The engine processes single natural language queries to identify user intent and extract parameters based on the provided configuration.

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
- **Text Generation**: Generates a basic textual summary and a raw data representation of the NLU results.
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
FR-2.2: The `run()` method shall process the input query using the configured NLU components (intent detector, parameter extractor, text generator).
FR-2.3: The `run()` method shall return a tuple containing:
    a) An NLU result dictionary including `intent`, `parameters`, `confidence`, `raw_response`, and `response_text`.
    b) A hint string (currently always `'new_conversation'`).
FR-2.4: Intent classification shall map the query to a command key defined in the metadata or `'unknown'`.
FR-2.5: Parameter extraction shall return a dictionary of identified parameter names and values.
FR-2.6: Text generation shall produce a structured raw response and a user-readable text summary.

## 4. Non-Functional Requirements

### 4.1 Usability
NFR-1.1: The `TalkEngine` API shall be simple and intuitive to use.
NFR-1.2: Configuration via dictionaries and optional overrides should be straightforward.

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
    print(f"  Confidence: {nlu_result.get('confidence', 0.0):.2f}")
    print(f"  Parameters: {nlu_result.get('parameters', {})}")
    # print(f"  Raw Response: {nlu_result.get('raw_response')}") # Internal data structure
    print(f"  Response Text: {nlu_result.get('response_text')}") # Basic text summary
``` 