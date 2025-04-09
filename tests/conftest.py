"""Shared test fixtures for talkengine tests."""

import shutil
from pathlib import Path
from typing import Union, Dict
import pytest
from unittest.mock import MagicMock

from talkengine.nlu_pipeline.nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    TextGenerationInterface,
)

# Import context and interaction models needed for fixtures/tests
from talkengine.models import NLUPipelineContext, InteractionState
from talkengine.nlu_pipeline.interaction_handlers import (
    ClarificationHandler,
    ValidationHandler,
    InteractionResult,
)


TMP_PATH_EXAMPLES: str = "./tests/tmp"


def _copy_directory(
    src_dir: Union[str, Path],
    dest_dir: Union[str, Path],
    exclude: list[str] | None = None,
) -> None:
    """Copy a directory to a destination.

    Args:
        src_dir: Source directory path
        dest_dir: Destination directory path
        exclude: list of file/directory names to exclude from copying
    """
    exclude = exclude or []
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    if not dest_path.exists():
        dest_path.mkdir(parents=True)

    for item in src_path.iterdir():
        if item.name in exclude:
            continue

        if item.is_dir():
            _copy_directory(item, dest_path / item.name, exclude)
        else:
            shutil.copy2(item, dest_path / item.name)


@pytest.fixture
def mock_intent_detector():
    mock = MagicMock(spec=IntentDetectionInterface)
    # Default return value for simple cases
    mock.classify_intent.return_value = {"intent": "mock_intent", "confidence": 0.95}
    return mock


@pytest.fixture
def mock_param_extractor():
    mock = MagicMock(spec=ParameterExtractionInterface)
    # Default return value: success, no validation needed
    mock.identify_parameters.return_value = (
        {},
        [],
    )  # Return tuple: (params, validation_requests)
    return mock


@pytest.fixture
def mock_text_generator():
    mock = MagicMock(spec=TextGenerationInterface)
    mock.generate_text.return_value = "mock_text"
    return mock


# Fixture for a basic NLUPipelineContext
@pytest.fixture
def default_pipeline_context():
    return NLUPipelineContext()


# Fixture for providing NLU overrides
@pytest.fixture
def mock_nlu_overrides(
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> Dict[
    str,
    Union[
        IntentDetectionInterface, ParameterExtractionInterface, TextGenerationInterface
    ],
]:
    return {
        "intent_detection": mock_intent_detector,
        "param_extraction": mock_param_extractor,
        "text_generation": mock_text_generator,
    }


# --- Fixtures for Mocking Interaction Handlers ---


@pytest.fixture
def mock_clarification_handler():
    mock = MagicMock(spec=ClarificationHandler)
    # Default initial prompt
    mock.get_initial_prompt.return_value = "Mock Clarification Prompt"
    # Default handling result (e.g., successful clarification)
    mock.handle_input.return_value = InteractionResult(
        response="Okay, clarified!",
        exit_mode=True,
        proceed_immediately=True,
        update_context={"current_intent": "clarified_intent"},
    )
    return mock


@pytest.fixture
def mock_validation_handler():
    mock = MagicMock(spec=ValidationHandler)
    mock.get_initial_prompt.return_value = "Mock Validation Prompt"
    mock.handle_input.return_value = InteractionResult(
        response="Okay, validated!",
        exit_mode=True,
        proceed_immediately=True,
        update_context={"current_parameters": {"validated_param": "valid_value"}},
    )
    return mock


# Fixture to provide mock interaction handlers to the engine
@pytest.fixture
def mock_interaction_handlers(
    mock_clarification_handler: MagicMock,
    mock_validation_handler: MagicMock,
) -> Dict[InteractionState, MagicMock]:
    return {
        InteractionState.CLARIFYING_INTENT: mock_clarification_handler,
        InteractionState.VALIDATING_PARAMETER: mock_validation_handler,
        # InteractionState.AWAITING_FEEDBACK: mock_feedback_handler,
    }
