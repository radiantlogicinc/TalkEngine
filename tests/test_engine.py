"""Tests for the TalkEngine class."""

import pytest
from unittest.mock import MagicMock, ANY
from typing import Dict, Any, List, Union

from talkengine import TalkEngine
from talkengine.nlu_pipeline.nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    TextGenerationInterface,
)
from talkengine.models import (
    NLUPipelineContext,
    NLUResult,
    ConversationDetail,
    InteractionState,
)

# Sample metadata for testing
TEST_METADATA = {
    "cmd1": {"description": "Command 1"},
    "cmd2": {"description": "Command 2", "parameters": {"p1": "str"}},
}


# --- Fixtures ---
@pytest.fixture
def mock_intent_detector():
    mock = MagicMock(spec=IntentDetectionInterface)
    mock.classify_intent.return_value = {"intent": "mock_intent", "confidence": 0.95}
    return mock


@pytest.fixture
def mock_param_extractor():
    mock = MagicMock(spec=ParameterExtractionInterface)
    mock.identify_parameters.return_value = ({"mock_param": "mock_value"}, [])
    return mock


@pytest.fixture
def mock_text_generator():
    mock = MagicMock(spec=TextGenerationInterface)
    mock.generate_text.return_value = "mock_text"
    return mock


@pytest.fixture
def mock_overrides(mock_intent_detector, mock_param_extractor, mock_text_generator):
    return {
        "intent_detection": mock_intent_detector,
        "param_extraction": mock_param_extractor,
        "text_generation": mock_text_generator,
    }


# --- Test Cases ---


def test_talkengine_init_defaults() -> None:
    """Test initializing TalkEngine with default NLU components."""
    engine = TalkEngine(command_metadata=TEST_METADATA)
    assert engine._command_metadata == TEST_METADATA
    assert engine._conversation_history == []
    # Check the stored configuration dict
    assert engine._nlu_overrides_config == {}
    assert isinstance(engine._intent_detector, IntentDetectionInterface)
    assert isinstance(engine._param_extractor, ParameterExtractionInterface)
    assert engine._text_generator is None or isinstance(
        engine._text_generator, TextGenerationInterface
    )
    # Check that defaults were instantiated (or add specific checks if needed)
    assert not engine._is_trained
    assert isinstance(engine._pipeline_context, NLUPipelineContext)
    assert engine._pipeline_context.interaction_mode == InteractionState.IDLE


def test_talkengine_init_with_history() -> None:
    """Test initializing TalkEngine with conversation history."""
    history = [{"role": "user", "content": "hi"}]
    engine = TalkEngine(command_metadata=TEST_METADATA, conversation_history=history)
    assert engine._conversation_history == history


def test_talkengine_init_with_overrides(
    mock_overrides: Dict[str, Any],
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test initializing TalkEngine with NLU overrides."""
    engine = TalkEngine(command_metadata=TEST_METADATA, nlu_overrides=mock_overrides)
    # Check the stored configuration dict
    assert engine._nlu_overrides_config == mock_overrides
    # Check that the correct override instances were assigned internally
    assert engine._intent_detector is mock_intent_detector
    assert engine._param_extractor is mock_param_extractor
    assert engine._text_generator is mock_text_generator
    assert isinstance(engine._pipeline_context, NLUPipelineContext)


def test_talkengine_train_placeholder() -> None:
    """Test the placeholder train() method."""
    engine = TalkEngine(command_metadata=TEST_METADATA)
    assert not engine._is_trained
    engine.train()  # Should run without error
    assert engine._is_trained


def test_talkengine_run_structure_defaults() -> None:
    """Test the structure of the run() method output with defaults (non-interactive path)."""
    engine = TalkEngine(command_metadata=TEST_METADATA)
    engine.train()
    result: NLUResult = engine.run("test query for cmd1")

    assert isinstance(result, NLUResult)
    # Default intent detector might find cmd1
    assert result.command == "cmd1"
    assert result.confidence is not None
    assert isinstance(result.parameters, dict)
    assert result.parameters == {}
    assert result.code_execution_result is None
    assert isinstance(result.conversation_detail, ConversationDetail)
    assert result.conversation_detail.interactions == []
    assert result.conversation_detail.response_text is None


def test_talkengine_run_with_overrides(
    mock_overrides: Dict[str, Any],
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test that run() calls the override methods (non-interactive path)."""
    engine = TalkEngine(command_metadata=TEST_METADATA, nlu_overrides=mock_overrides)
    engine.train()
    query = "another test query"

    # Configure mocks for a successful non-interactive run
    mock_intent_detector.classify_intent.return_value = {
        "intent": "mock_intent",
        "confidence": 0.99,
    }
    mock_params = {"mock_param": "mock_value"}
    mock_param_extractor.identify_parameters.return_value = (mock_params, [])
    mock_text = "mock_text"
    mock_text_generator.generate_text.return_value = mock_text

    result: NLUResult = engine.run(query)

    assert isinstance(result, NLUResult)
    assert result.command == "mock_intent"
    assert result.parameters == mock_params
    assert result.confidence == 0.99
    assert result.code_execution_result is None
    assert isinstance(result.conversation_detail, ConversationDetail)
    assert result.conversation_detail.interactions == []
    assert result.conversation_detail.response_text == mock_text

    # Check if the correct mock instances were called with context and correct args
    # ANY used for context as its state changes during run
    mock_intent_detector.classify_intent.assert_called_once_with(query, ANY, None)
    mock_param_extractor.identify_parameters.assert_called_once_with(
        query, "mock_intent", ANY
    )
    mock_text_generator.generate_text.assert_called_once_with(
        "mock_intent", mock_params, None, ANY
    )


def test_talkengine_reset() -> None:
    """Test resetting the engine."""
    history1 = [{"role": "user", "content": "q1"}]
    engine = TalkEngine(command_metadata=TEST_METADATA, conversation_history=history1)
    engine.train()
    assert engine._command_metadata == TEST_METADATA
    assert engine._conversation_history == history1
    assert engine._is_trained
    initial_intent_detector = engine._intent_detector  # Keep ref to initial instance

    new_metadata = {"new_cmd": {"description": "New"}}
    new_history: List[Dict[str, Any]] = []
    engine.reset(command_metadata=new_metadata, conversation_history=new_history)

    assert engine._command_metadata == new_metadata
    assert engine._conversation_history == new_history
    # Check if components are re-initialized (e.g., intent detector has new meta)
    # Assuming default detector is used after reset without overrides
    assert isinstance(engine._intent_detector, IntentDetectionInterface)
    assert engine._intent_detector is not initial_intent_detector  # Ensure new instance
    assert isinstance(engine._pipeline_context, NLUPipelineContext)
    assert engine._pipeline_context.interaction_mode == InteractionState.IDLE
    # train() needs to be called again after reset
    assert not engine._is_trained


# TODO: Add tests for default implementation logic once specified.
# TODO: Add tests for error handling (e.g., invalid overrides).

# --- New Tests for Refactored Engine ---


def test_talkengine_run_excluded_intents(
    mock_overrides: Dict[str, Any], mock_intent_detector: MagicMock
) -> None:
    """Test run() correctly passes excluded_intents to intent detector."""
    engine = TalkEngine(command_metadata=TEST_METADATA, nlu_overrides=mock_overrides)
    engine.train()
    query = "some query"
    exclude = ["cmd1"]

    engine.run(query, excluded_intents=exclude)

    # Check intent detector was called with the exclusion list
    mock_intent_detector.classify_intent.assert_called_once_with(query, ANY, exclude)


def test_talkengine_run_with_code_execution(
    mock_overrides: Dict[str, Any],
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test run() correctly executes code specified in metadata."""
    mock_executable = MagicMock(return_value={"code_ran": True})
    metadata_with_code = {
        "code_cmd": {
            "description": "Command with code",
            "parameters": {"arg": "int"},
            "executable_code": mock_executable,
        }
    }
    engine = TalkEngine(
        command_metadata=metadata_with_code, nlu_overrides=mock_overrides
    )
    engine.train()
    query = "run code cmd with arg 5"
    mock_params = {"arg": 5}

    # Configure mocks
    mock_intent_detector.classify_intent.return_value = {
        "intent": "code_cmd",
        "confidence": 0.99,
    }
    mock_param_extractor.identify_parameters.return_value = (mock_params, [])
    mock_text_generator.generate_text.return_value = "Code executed and text generated."

    result: NLUResult = engine.run(query)

    # Assert that the executable was called
    mock_executable.assert_called_once_with(mock_params)

    # Assert the result includes the code execution output
    assert isinstance(result, NLUResult)
    assert result.command == "code_cmd"
    assert result.parameters == mock_params
    assert result.code_execution_result == {"code_ran": True}
    assert (
        result.conversation_detail.response_text == "Code executed and text generated."
    )

    # Assert text generator received the code result
    mock_text_generator.generate_text.assert_called_once_with(
        "code_cmd", mock_params, {"code_ran": True}, ANY
    )


def test_talkengine_run_only_code_execution(
    mock_overrides: Dict[str, Any],
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
) -> None:
    """Test run() when only code execution is configured (no text gen)."""
    mock_executable = MagicMock(return_value={"code_ran": True})
    metadata_with_code = {
        "code_cmd": {"description": "Code only", "executable_code": mock_executable}
    }
    # Add specific type hint for this dictionary
    overrides_no_text: Dict[
        str, Union[IntentDetectionInterface, ParameterExtractionInterface]
    ] = {
        "intent_detection": mock_intent_detector,
        "param_extraction": mock_param_extractor,
    }
    engine = TalkEngine(
        command_metadata=metadata_with_code, nlu_overrides=overrides_no_text
    )
    engine.train()
    query = "run code only cmd"
    mock_params: Dict[str, Any] = {}

    mock_intent_detector.classify_intent.return_value = {
        "intent": "code_cmd",
        "confidence": 0.99,
    }
    mock_param_extractor.identify_parameters.return_value = (mock_params, [])

    result: NLUResult = engine.run(query)

    mock_executable.assert_called_once_with(mock_params)
    assert result.command == "code_cmd"
    assert result.code_execution_result == {"code_ran": True}
    assert result.conversation_detail.response_text is None  # No text generator


def test_talkengine_run_only_text_generation(
    mock_overrides: Dict[str, Any],
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test run() when only text generation is configured (no code exec)."""
    metadata_no_code = {"text_cmd": {"description": "Text only"}}
    engine = TalkEngine(command_metadata=metadata_no_code, nlu_overrides=mock_overrides)
    engine.train()
    query = "run text only cmd"
    mock_params: Dict[str, Any] = {}

    mock_intent_detector.classify_intent.return_value = {
        "intent": "text_cmd",
        "confidence": 0.99,
    }
    mock_param_extractor.identify_parameters.return_value = (mock_params, [])
    mock_text_generator.generate_text.return_value = "Only text generated."

    result: NLUResult = engine.run(query)

    assert result.command == "text_cmd"
    assert result.code_execution_result is None  # No code executed
    assert result.conversation_detail.response_text == "Only text generated."
    mock_text_generator.generate_text.assert_called_once_with(
        "text_cmd", mock_params, None, ANY  # Code result is None
    )


def test_talkengine_run_no_code_or_text(
    mock_overrides: Dict[str, Any],
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
) -> None:
    """Test run() when neither code exec nor text gen is configured."""
    metadata_minimal = {"minimal_cmd": {"description": "Minimal"}}
    # Add specific type hint for this dictionary
    overrides_no_text: Dict[
        str, Union[IntentDetectionInterface, ParameterExtractionInterface]
    ] = {
        "intent_detection": mock_intent_detector,
        "param_extraction": mock_param_extractor,
    }
    engine = TalkEngine(
        command_metadata=metadata_minimal, nlu_overrides=overrides_no_text
    )
    engine.train()
    query = "run minimal cmd"
    mock_params: Dict[str, Any] = {}

    mock_intent_detector.classify_intent.return_value = {
        "intent": "minimal_cmd",
        "confidence": 0.99,
    }
    mock_param_extractor.identify_parameters.return_value = (mock_params, [])

    result: NLUResult = engine.run(query)

    assert result.command == "minimal_cmd"
    assert result.code_execution_result is None
    assert result.conversation_detail.response_text is None
