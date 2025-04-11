# talkengine - Software Requirements Specification (Streamlined)

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for the streamlined `talkengine` library, focusing on providing a simple NLU pipeline via the `TalkEngine` class.

### 1.2 Product Scope
`talkengine` allows developers to initialize an NLU pipeline (`TalkEngine`) with command descriptions (metadata defining command parameters via Pydantic `BaseModel` classes), optional conversation history, and optional NLU component overrides (which can also include command execution logic). The engine processes natural language queries, potentially engaging in interactive sub-dialogues (e.g., for clarification or validation) to identify user intent and extract parameters based on the provided configuration.

### 1.3 Intended Audience
This document is intended for developers using the `talkengine` library.

## 2. Overall Description

### 2.1 Product Perspective
`talkengine` is a lightweight Python library providing a configurable NLU pipeline object (`TalkEngine`) for basic intent classification and parameter extraction from single user queries.

### 2.2 Product Features
- **Configurable Pipeline Initialization**: `TalkEngine` is initialized directly with `command_metadata` (containing command description and a reference to a Pydantic `BaseModel` class defining parameters), optional `conversation_history`, and optional `nlu_overrides`.
- **NLU Overrides**: `nlu_overrides` allows specifying custom NLU component implementations (intent detection, parameter extraction, text generation).
- **Command Code Execution (via Overrides)**: `nlu_overrides` can optionally define the executable code (`function` and expected Pydantic `BaseModel` `result_class`) for specific commands.
- **Single Query Processing**: The `run()` method processes one user query at a time, returning a single `NLUResult` object.
- **Intent Classification**: Maps user input to a predefined command key based on the provided metadata (`description` field) and configured intent detector.
- **Parameter Extraction**: Identifies and extracts parameter values based on the user input, classified intent, and the fields defined in the `parameter_class` specified in the command's metadata.
- **Interactive Sub-dialogues**: Can enter specific modes to:
    - Clarify ambiguous user intent.
    - Validate or request missing parameters.
    - Solicit feedback on previous responses.
- **Code Execution**: If configured for a command in `nlu_overrides`, executes the specified function with parameters populated into an instance of the command's `parameter_class`. The result (an instance of the specified `result_class`) is captured.
- **Text Generation**: Generates a textual response based on the NLU results (command, parameters, and potentially the result object from code execution). Can use a default generator or a custom one provided via `nlu_overrides`.
- **Stateful Reset**: The `reset()` method allows re-initializing the `TalkEngine` instance with new configuration.

### 2.3 User Classes and Characteristics
Python developers needing a simple, configurable NLU pipeline for single-turn query processing.

### 2.4 Operating Environment
- Python 3.12+
- Dependencies: (Check `pyproject.toml` - currently primarily standard libraries + `pytest` for dev)

## 3. Functional Requirements

### 3.1 `TalkEngine` Initialization & Reset
FR-1.1: The system shall provide a `TalkEngine` class.
FR-1.2: `TalkEngine` shall be initializable with a `command_metadata` dictionary. This dictionary maps command name strings to definition objects/dictionaries containing:
    a) `description` (str): Natural language description of the command.
    b) `parameter_class` (Type[BaseModel]): A reference to the Pydantic `BaseModel` subclass defining the parameters for the command.
FR-1.3: `TalkEngine` shall accept optional `conversation_history` (list of dicts) during initialization.
FR-1.4: `TalkEngine` shall accept an optional `nlu_overrides` dictionary during initialization. This dictionary can contain:
    a) Mappings from NLU interface names (e.g., `"intent_detection"`) to custom implementation instances/functions.
    b) Mappings from command name strings to override configurations for that specific command. Currently, this supports an `"executable_code"` key, whose value is a dictionary containing:
        i) `function` (Callable): The Python function to execute for this command.
        ii) `result_class` (Type[BaseModel]): The Pydantic `BaseModel` subclass expected as the return type from the `function`.
FR-1.5: `TalkEngine` shall provide a `reset()` method to re-initialize the instance with new `command_metadata`, `conversation_history`, and `nlu_overrides`, adhering to the structures defined in FR-1.2, FR-1.3, and FR-1.4.
FR-1.6: `TalkEngine` shall provide a placeholder `train()` method.

### 3.2 NLU Processing (`run` method)
FR-2.1: `TalkEngine` shall provide a `run(query: str)` method.
FR-2.2: The `run()` method shall process the input query using the configured NLU components (intent detector, parameter extractor) and internal logic (potentially including code execution as defined in `nlu_overrides` and text generation).
FR-2.3: The `run()` method shall return a single `NLUResult` object representing the outcome of the NLU pipeline processing for the given query.
FR-2.4: The `NLUResult` object shall be a Pydantic BaseModel containing:
    a) `command` (Optional[str]): The identified command key from the metadata, `'unknown'`, or `None` if processing is interrupted (e.g., by interaction).
    b) `parameters` (Optional[dict]): A dictionary of extracted parameter names and values, corresponding to the fields defined in the command's `parameter_class` (from `command_metadata`). Populated by the parameter extraction component.
    c) `artifacts` (Optional[BaseModel]): An instance of the Pydantic `BaseModel` specified by the `result_class` in the command's `executable_code` override (within `nlu_overrides`), if code was executed. `None` otherwise.
    d) `conversation_detail` (ConversationDetail): An object containing details about the interaction flow and response text for this attempt (See FR-2.5).
FR-2.5: The `ConversationDetail` object within `NLUResult` shall be a Pydantic BaseModel containing:
    a) `interactions` (list): A list of tuples or objects, each documenting a step in any interactive sub-dialogue (e.g., clarification, validation). Each entry should record the interaction stage/type, the prompt presented to the user, and the user's feedback/response during that interaction step.
    b) `response_text` (Optional[str]): The final user-facing text generated for this attempt (or the prompt if interaction is ongoing).
FR-2.6: Intent classification shall identify a command key defined in the metadata (or `'unknown'`), contributing to the `command` field of the `NLUResult`. The default implementation utilizes the command `description` field from the metadata for this classification.
FR-2.7: Parameter extraction shall identify parameter names and values based on the input query, the classified intent, and the fields defined within the `parameter_class` associated with the command in `command_metadata`. It populates the `parameters` dictionary field of the `NLUResult`.
FR-2.8: The response generation phase involves two potential steps orchestrated by the `TalkEngine`:
    a) Code Execution: If an `executable_code` entry (containing `function` and `result_class`) exists for the identified command key within `nlu_overrides`, the `TalkEngine` shall:
        i. Attempt to instantiate the command's `parameter_class` (from `command_metadata`) using the extracted `parameters` dictionary.
        ii. Call the specified `function`, passing the instantiated parameter object.
        iii. Validate that the function's return value is an instance of the specified `result_class`.
        iv. Store the returned `result_class` instance in the `artifacts` field of the `NLUResult`.
    b) Text Generation: If a text generator component is configured (via `nlu_overrides` or using the default), the `TalkEngine` shall invoke it to generate a user-facing response. This component receives context including the `command`, `parameters` (dict), and `artifacts` (the `BaseModel` instance from code execution, if any). The result populates the `response_text` field of the `ConversationDetail`.
    c) Configuration Requirements: Every command defined in `command_metadata` must have a `description` and a `parameter_class`. Defining `executable_code` in `nlu_overrides` is optional per command. Using a custom text generator via `nlu_overrides` is optional; a default generator may be used otherwise. Text generation will occur even if code execution does not.
FR-2.9: The `run` method shall accept an optional argument `excluded_intents` (list of strings) to exclude specific command intents from the classification step. This allows a calling application to re-invoke `run` if initial results are unsatisfactory.

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
NFR-1.2: Configuration via `command_metadata` (using Pydantic classes for parameters) and `nlu_overrides` (for NLU components and command execution) should be straightforward.
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
from talkengine.nlu_pipeline.models import NLUResult # Import NLUResult
from typing import Optional # For executable example
from pydantic import BaseModel # Import BaseModel

# Configure logging to see internal steps
logging.basicConfig(level=logging.INFO)

# 1. Define your command metadata (using Pydantic models)
class CommandParameters(BaseModel):
    num1: int
    num2: int

class CommandResult(BaseModel):
    result: int
    success: bool
    error: Optional[str] = None

command_metadata = {
    "add_numbers": {
        "description": "Add two numbers together.",
        "parameter_class": CommandParameters,
    },
}

# 2. (Optional) Define conversation history
conversation_history = []

# 3. (Optional) Define NLU overrides, including executable code
def add_numbers(input_params: CommandParameters) -> CommandResult:
    \"\"\"Add two numbers together.\"\"\"
    try:
        result = input_params.num1 + input_params.num2
        return CommandResult(
            result=result,
            success=True,
            error=None
        )
    except ValueError:
        return CommandResult(
            result=-999,
            success=False,
            error="Invalid number input"
        )

nlu_overrides = {
    "add_numbers": {
        "executable_code": {
            "function": add_numbers,
            "result_class": CommandResult,
        },
    },
    # Add other overrides for intent detection, param extraction, text gen here if needed
    # "intent_detection": MyCustomIntentDetector(),
}

# 4. Initialize the engine
engine = TalkEngine(
    command_metadata=command_metadata,
    conversation_history=conversation_history,
    nlu_overrides=nlu_overrides
)

# 5. Train the engine (currently a placeholder)
engine.train()

# 6. Process queries in a loop
print("\\nTalkEngine Ready. Type 'quit' to exit.")
while True:
    user_query = input("> ")
    if user_query.lower() == 'quit':
        break

    # engine.run() now returns a single NLUResult object
    # excluded_intents can be optionally passed if needed
    result: NLUResult = engine.run(user_query)

    # Check if the engine is prompting for clarification or validation
    # (Assuming interaction logic updates result fields or uses a specific state)
    # Example check (replace with actual interaction handling check):
    # if result.interaction_needed:
    #     print(f"  Interaction Required: {result.interaction_prompt}")
    #     continue

    # --- Display final NLU results ---
    print(f"  Command: {result.command or 'N/A'}")
    print(f"  Parameters: {result.parameters or {}}")
    # Check if code was executed and artifacts (BaseModel instance) exist
    if result.artifacts:
        print(f"  Artifacts: {result.artifacts}") # Pydantic model auto-repr
    # Check the generated response text (if any)
    print(f"  Response Text: {result.conversation_detail.response_text}")
    # You can also inspect interactions if needed
    # print(f"  Interactions Log: {result.conversation_detail.interactions}")

# Example of resetting the engine
# new_metadata = { ... }
# new_overrides = { ... }
# engine.reset(command_metadata=new_metadata, nlu_overrides=new_overrides)
# engine.train()
# result = engine.run("new query")
``` 