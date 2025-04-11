"""Shared test fixtures for talkengine tests."""

import logging
from typing import Optional

import pytest
from unittest.mock import MagicMock

from pydantic import BaseModel

from talkengine import TalkEngine
from talkengine.nlu_pipeline.nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    TextGenerationInterface,
)

# Import context and interaction models needed for fixtures/tests
from talkengine.nlu_pipeline.models import NLUPipelineContext, InteractionState
from talkengine.nlu_pipeline.interaction_handlers import (
    ClarificationHandler,
    ValidationHandler,
)

# Import types for fixtures
from talkengine.types import NLUOverridesConfig, CommandMetadataConfig

# Configure basic logging for testing (so caplog works)
logging.basicConfig(level=logging.DEBUG)  # Or INFO/ERROR as needed

TMP_PATH_EXAMPLES: str = "./tests/tmp"


# --- Dummy Models for Testing --- #
class DummyParams(BaseModel):
    param1: str = "default_value"
    param2: Optional[int] = None
    required_param: float  # A required parameter


class SimpleParams(BaseModel):
    # No fields, or only optional ones
    pass


class DummyResult(BaseModel):
    output: str
    status_code: int


# --- Dummy Functions for Testing --- #
def dummy_exec_func_success(params: DummyParams) -> DummyResult:
    """Dummy function that 'succeeds'."""
    return DummyResult(output=f"Processed {params.param1}", status_code=200)


def dummy_exec_func_fail(params: DummyParams) -> DummyResult:
    """Dummy function that 'fails' (raises exception)."""
    raise ValueError("Dummy execution failed")


def dummy_exec_func_wrong_type(params: DummyParams) -> dict:
    """Dummy function that returns the wrong type."""
    return {"wrong": "type"}


# --- Fixtures for Command Metadata --- #
@pytest.fixture
def valid_command_metadata() -> CommandMetadataConfig:
    """Provides valid command metadata for testing."""
    return {
        "test_command": {
            "description": "A test command with params and potential execution.",
            "parameter_class": DummyParams,
        },
        "other_command": {
            "description": "Another command description.",
            "parameter_class": DummyParams,  # Can reuse or use a different one
        },
        "simple_command": {  # Add a simple command
            "description": "A simple command with no required params.",
            "parameter_class": SimpleParams,
        },
    }


@pytest.fixture
def invalid_command_metadata_missing_desc() -> dict:
    return {"test_command": {"parameter_class": DummyParams}}


@pytest.fixture
def invalid_command_metadata_missing_param_class() -> dict:
    return {"test_command": {"description": "A test command."}}


@pytest.fixture
def invalid_command_metadata_param_not_basemodel() -> dict:
    class NotBaseModel:
        pass

    return {"test_command": {"description": "desc", "parameter_class": NotBaseModel}}


# --- Fixtures for NLU Overrides --- #


@pytest.fixture
def empty_nlu_overrides() -> NLUOverridesConfig:
    return {}


@pytest.fixture
def valid_nlu_overrides_components_only(
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> NLUOverridesConfig:
    """Overrides with only NLU component mocks."""
    return {
        "intent_detection": mock_intent_detector,
        "param_extraction": mock_param_extractor,
        "text_generation": mock_text_generator,
    }


@pytest.fixture
def valid_nlu_overrides_exec_code_only() -> NLUOverridesConfig:
    """Overrides with only executable code."""
    return {
        "test_command": {
            "executable_code": {
                "function": dummy_exec_func_success,
                "result_class": DummyResult,
            }
        }
    }


@pytest.fixture
def valid_nlu_overrides_mixed(
    mock_intent_detector: MagicMock,
) -> NLUOverridesConfig:
    """Overrides with a mix of components and executable code."""
    return {
        "intent_detection": mock_intent_detector,
        "other_command": {
            "executable_code": {
                "function": dummy_exec_func_success,  # Can reuse
                "result_class": DummyResult,
            }
        },
    }


@pytest.fixture
def invalid_nlu_overrides_bad_exec_structure() -> dict:
    return {"test_command": {"executable_code": "not_a_dict"}}


@pytest.fixture
def invalid_nlu_overrides_missing_function() -> dict:
    return {"test_command": {"executable_code": {"result_class": DummyResult}}}


@pytest.fixture
def invalid_nlu_overrides_missing_result_class() -> dict:
    return {"test_command": {"executable_code": {"function": dummy_exec_func_success}}}


@pytest.fixture
def invalid_nlu_overrides_non_callable_func() -> dict:
    return {
        "test_command": {
            "executable_code": {"function": 123, "result_class": DummyResult}
        }
    }


@pytest.fixture
def invalid_nlu_overrides_result_not_basemodel() -> dict:
    class NotBaseModel:
        pass

    return {
        "test_command": {
            "executable_code": {
                "function": dummy_exec_func_success,
                "result_class": NotBaseModel,
            }
        }
    }


# --- Fixtures for NLU Components (Mocks) --- #


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
    # Now returns (params_dict, validation_requests_list)
    mock.identify_parameters.return_value = ({}, [])
    return mock


@pytest.fixture
def mock_text_generator():
    mock = MagicMock(spec=TextGenerationInterface)
    mock.generate_text.return_value = "mock_text"
    return mock


@pytest.fixture
def default_pipeline_context():
    # This might need adjustment if NLUPipelineContext init changes significantly
    # It now takes command_metadata, let's provide a minimal valid one
    return NLUPipelineContext(
        command_metadata={  # Provide minimal valid meta
            "dummy": {"description": "d", "parameter_class": DummyParams}
        }
    )


# --- Fixtures for Mocking Interaction Handlers --- #


@pytest.fixture
def mock_clarification_handler():
    mock = MagicMock(spec=ClarificationHandler)
    # Default initial prompt
    mock.get_initial_prompt.return_value = "Mock Clarification Prompt"
    # Default handling result (using tuple format)
    mock_context = MagicMock(spec=NLUPipelineContext)
    mock.handle_input.return_value = (
        mock_context,
        True,
        None,
    )  # Default: proceed, no specific next step
    return mock


@pytest.fixture
def mock_validation_handler():
    """Fixture for a mocked ValidationHandler."""
    mock = MagicMock(spec=ValidationHandler)
    # Default initial prompt
    mock.get_initial_prompt.return_value = "Mock Validation Prompt"
    # Default handling result (using tuple format)
    mock_context = MagicMock(spec=NLUPipelineContext)
    mock.handle_input.return_value = (
        mock_context,
        True,
        None,
    )  # Default: proceed, no specific next step
    return mock


# Fixture to provide mock interaction handlers to the engine
@pytest.fixture
def mock_interaction_handlers(
    mock_clarification_handler: MagicMock,
    mock_validation_handler: MagicMock,
) -> dict[InteractionState, MagicMock]:
    return {
        InteractionState.CLARIFYING_INTENT: mock_clarification_handler,
        InteractionState.VALIDATING_PARAMETER: mock_validation_handler,
        # InteractionState.AWAITING_FEEDBACK: mock_feedback_handler,
    }


# --- Fixture for TalkEngine Instance --- #


@pytest.fixture
def talk_engine_instance(
    valid_command_metadata: CommandMetadataConfig,
    valid_nlu_overrides_components_only: NLUOverridesConfig,
) -> TalkEngine:
    """Provides a TalkEngine instance with valid basic config and mock components."""
    return TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
