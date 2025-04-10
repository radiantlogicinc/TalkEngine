# talkengine

A simple NLU pipeline library for Python.

<p align="center">
  <img src="assets/logo.png" alt="talkengine Logo" width="200"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/talkengine/"><img alt="PyPI" src="https://img.shields.io/pypi/v/talkengine"></a>
  <a href="https://github.com/username/talkengine/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/username/talkengine"></a>
  <a href="https://github.com/username/talkengine/actions"><img alt="Build Status" src="https://img.shields.io/github/workflow/status/username/talkengine/tests"></a>
  <a href="https://github.com/username/talkengine/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/username/talkengine"></a>
</p>

## 📋 Overview

talkengine provides a straightforward way to process natural language queries against a predefined set of commands. You initialize the `TalkEngine` class with your command descriptions and can then process user queries to identify intent and parameters.

## ✨ Features

- **Simple Intent Classification**: Maps user input to predefined command keys based on metadata.
- **Basic Parameter Extraction**: Identifies potential parameters (default implementation is basic).
- **Configurable NLU**: Allows providing custom functions (overrides) for intent detection, parameter extraction, and response generation.
- **Stateful Reset**: Engine can be reset and re-initialized with new command metadata.

## 🚀 Installation

```bash
pip install talkengine
```

## 🏁 Quick Start

```python
import logging
from talkengine import TalkEngine
from talkengine.nlu_pipeline.models import NLUResult # Import NLUResult
from typing import Optional # For executable example

# Configure logging to see internal steps
logging.basicConfig(level=logging.INFO)

# 1. Define your command metadata
def add_numbers(num1: int, num2: int) -> int:
    """Add two numbers together."""
    try:
        return num1 + num2
    except ValueError:
        return {"error": "Invalid number input"}

command_metadata = {
    "add_numbers": {
        "description": "Add two numbers together.",
        "parameters": {"num1": "int", "num2": "int"},
    },
}

# 2. (Optional) Define conversation history
conversation_history = [
    {"role": "user", "content": "What was the weather yesterday?"},
    # Add previous NLU results if available for context
]

# 3. (Optional) Define NLU overrides
nlu_overrides = {
    "add_numbers": {
        "executable_code": add_numbers # Add reference to executable function
    },
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
print("\nTalkEngine Ready. Type 'quit' to exit.")
while True:
    user_query = input("> ")
    if user_query.lower() == 'quit':
        break

    # engine.run() now returns a single NLUResult object
    # excluded_intents can be optionally passed if needed
    result: NLUResult = engine.run(user_query)

    # Check if the engine is prompting for clarification or validation
    # The context object inside the engine tracks the interaction mode
    if engine._pipeline_context.interaction_mode != InteractionState.IDLE:
        print(f"  Interaction Required ({engine._pipeline_context.interaction_mode.name}):")
        # The prompt is now in the response_text field of conversation_detail
        print(f"  Prompt: {result.conversation_detail.response_text}")
        continue # Skip printing normal results for this example

    # --- Display final NLU results --- 
    print(f"  Command: {result.command or 'N/A'}")
    print(f"  Parameters: {result.parameters or {}}")
    # Check if code was executed
    if result.artifacts:
        print(f"  Artifacts: {result.artifacts}")
    # Check the generated response text (if any)
    print(f"  Response Text: {result.conversation_detail.response_text}")
    # You can also inspect interactions if needed
    # print(f"  Interactions Log: {result.conversation_detail.interactions}")

# Example of resetting the engine
# new_metadata = { ... }
# engine.reset(command_metadata=new_metadata)
# engine.train()
# result = engine.run("new query")

```

## 📚 API Reference

Key class: `talkengine.TalkEngine`

- `__init__(command_metadata, conversation_history=None, nlu_overrides=None)`: Initializes the engine.
- `train()`: Placeholder for potential future training/configuration steps.
- `run(query, excluded_intents=None)`: Processes a query, returns `NLUResult` object.
- `reset(command_metadata, conversation_history=None, nlu_overrides=None)`: Re-initializes the engine.

`NLUResult` Object Attributes:
- `command` (Optional[str]): Identified command key (or `None` during interaction, `'unknown'` if not found).
- `parameters` (Optional[Dict]): Extracted parameters.
- `artifacts` (Optional[Dict]): Dictionary result from executed code, if any.
- `conversation_detail` (ConversationDetail): Contains interaction log and final response text.

`ConversationDetail` Object Attributes:
- `interactions` (List[Tuple[str, str, Optional[str]]]): Log of (stage, prompt, user_response) tuples during interactions.
- `response_text` (Optional[str]): Final user-facing response text, or the interaction prompt if interaction is ongoing.

## 🛠️ Development

### Prerequisites

- Python 3.12+
- uv (optional, for dependency management)

### Setup for Development

```bash
# Clone the repository
git clone https://github.com/username/talkengine.git
cd talkengine

# (Optional) Install uv
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment (using venv or uv)
# python -m venv .venv  OR  uv venv

# Activate the virtual environment
# source .venv/bin/activate  OR  source .venv/bin/activate

# Install development dependencies
# pip install -e ".[dev]"  OR  uv pip install -e ".[dev]"
```

### Running Tests
```bash
# Install pytest if needed (pip install pytest or uv pip install pytest)
pytest
```

## 🤝 Contributing

Contributions are welcome! Please check [CONTRIBUTING.md](CONTRIBUTING.md) (if it exists) or open an issue/PR.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Thanks to all contributors.
