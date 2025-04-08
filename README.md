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
- **Configurable NLU**: Allows providing custom functions (overrides) for intent detection, parameter extraction, and text generation.
- **Stateful Reset**: Engine can be reset and re-initialized with new command metadata.

## 🚀 Installation

```bash
pip install talkengine
```

## 🏁 Quick Start

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

    # Note: If the engine needs clarification or parameter validation,
    # the nlu_result dictionary will contain an 'interaction_prompt'
    # key instead of 'intent', 'parameters', etc.
    # The 'hint' will indicate the interaction state (e.g., 'awaiting_clarification').
    # Your application logic should handle these interaction prompts.

    print(f"  Intent: {nlu_result.get('intent', 'N/A')}")
    print(f"  Confidence: {nlu_result.get('confidence', 0.0):.2f}")
    print(f"  Parameters: {nlu_result.get('parameters', {})}")
    # print(f"  Raw Response: {nlu_result.get('raw_response')}") # Internal data structure
    print(f"  Response Text: {nlu_result.get('response_text')}") # Basic text summary

# Example of resetting the engine
# new_metadata = { ... }
# engine.reset(command_metadata=new_metadata)
# engine.train()
# result, hint = engine.run("new query")

```

## 📚 API Reference

Key class: `talkengine.TalkEngine`

- `__init__(command_metadata, conversation_history=None, nlu_overrides=None)`: Initializes the engine.
- `train()`: Placeholder for potential future training/configuration steps.
- `run(query)`: Processes a query, returns `(nlu_result_dict, hint_str)`.
- `reset(command_metadata, conversation_history=None, nlu_overrides=None)`: Re-initializes the engine.

NLU Result Dictionary Keys:
- `intent` (str): Identified command key (or 'unknown').
- `parameters` (dict): Extracted parameters.
- `confidence` (float): Confidence score for the intent classification.
- `raw_response` (Any): Internal data structure from text generation.
- `response_text` (str): Basic textual summary of the NLU result.

Interaction Prompt Dictionary Keys (when hint indicates interaction):
- `interaction_prompt` (str): The prompt message to show the user.
- `interaction_mode` (InteractionState): The specific mode the engine is in.

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
