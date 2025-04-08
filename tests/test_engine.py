"""Tests for the TalkEngine class."""

import pytest
from unittest.mock import MagicMock

from talkengine import TalkEngine
from talkengine.nlu_pipeline.nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    ResponseGenerationInterface,
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
    mock.identify_parameters.return_value = {"mock_param": "mock_value"}
    return mock


@pytest.fixture
def mock_text_generator():
    mock = MagicMock(spec=ResponseGenerationInterface)
    mock.generate_response.return_value = ({"mock_raw": True}, "mock_text")
    return mock


@pytest.fixture
def mock_overrides(mock_intent_detector, mock_param_extractor, mock_text_generator):
    # This fixture now represents the input dictionary for nlu_overrides
    return {
        "intent_detection": mock_intent_detector,
        "param_extraction": mock_param_extractor,
        "text_generation": mock_text_generator,
    }


# --- Test Cases ---


def test_talkengine_init_defaults():
    """Test initializing TalkEngine with default NLU components."""
    engine = TalkEngine(command_metadata=TEST_METADATA)
    assert engine._command_metadata == TEST_METADATA
    assert engine._conversation_history == []
    # Check the stored configuration dict
    assert engine._nlu_overrides_config == {}
    assert isinstance(engine._intent_detector, IntentDetectionInterface)
    assert isinstance(engine._param_extractor, ParameterExtractionInterface)
    assert isinstance(engine._text_generator, ResponseGenerationInterface)
    # Check that defaults were instantiated (or add specific checks if needed)
    assert not engine._is_trained


def test_talkengine_init_with_history():
    """Test initializing TalkEngine with conversation history."""
    history = [{"role": "user", "content": "hi"}]
    engine = TalkEngine(command_metadata=TEST_METADATA, conversation_history=history)
    assert engine._conversation_history == history


def test_talkengine_init_with_overrides(
    mock_overrides, mock_intent_detector, mock_param_extractor, mock_text_generator
):
    """Test initializing TalkEngine with NLU overrides."""
    engine = TalkEngine(command_metadata=TEST_METADATA, nlu_overrides=mock_overrides)
    # Check the stored configuration dict
    assert engine._nlu_overrides_config == mock_overrides
    # Check that the correct override instances were assigned internally
    assert engine._intent_detector is mock_intent_detector
    assert engine._param_extractor is mock_param_extractor
    assert engine._text_generator is mock_text_generator


def test_talkengine_train_placeholder():
    """Test the placeholder train() method."""
    engine = TalkEngine(command_metadata=TEST_METADATA)
    assert not engine._is_trained
    engine.train()  # Should run without error
    assert engine._is_trained


def test_talkengine_run_structure_defaults():
    """Test the structure of the run() method output with defaults."""
    engine = TalkEngine(command_metadata=TEST_METADATA)
    engine.train()
    result, hint = engine.run("test query")

    assert hint == "new_conversation"
    assert isinstance(result, dict)
    assert "intent" in result
    assert "parameters" in result
    assert "confidence" in result
    assert "raw_response" in result
    assert "response_text" in result
    # Default impls might return specific values, can test those later


def test_talkengine_run_with_overrides(mock_overrides):
    """Test that run() calls the override methods."""
    engine = TalkEngine(command_metadata=TEST_METADATA, nlu_overrides=mock_overrides)
    engine.train()
    query = "another test query"
    result, hint = engine.run(query)

    # Check return structure matches what mocks provide
    assert hint == "new_conversation"
    assert result["intent"] == "mock_intent"
    assert result["parameters"] == {"mock_param": "mock_value"}
    assert result["confidence"] == 0.95
    assert result["raw_response"] == {"mock_raw": True}
    assert result["response_text"] == "mock_text"

    # Check if the *correct mock instances* stored inside the engine were called
    engine._intent_detector.classify_intent.assert_called_once_with(query)
    engine._param_extractor.identify_parameters.assert_called_once_with(
        query, "mock_intent"
    )
    engine._text_generator.generate_response.assert_called_once_with(
        "mock_intent", {"mock_param": "mock_value"}
    )


def test_talkengine_reset():
    """Test resetting the engine."""
    history1 = [{"role": "user", "content": "q1"}]
    engine = TalkEngine(command_metadata=TEST_METADATA, conversation_history=history1)
    engine.train()
    assert engine._command_metadata == TEST_METADATA
    assert engine._conversation_history == history1
    assert engine._is_trained
    initial_intent_detector = engine._intent_detector  # Keep ref to initial instance

    new_metadata = {"new_cmd": {"description": "New"}}
    new_history = []
    engine.reset(command_metadata=new_metadata, conversation_history=new_history)

    assert engine._command_metadata == new_metadata
    assert engine._conversation_history == new_history
    # Check if components are re-initialized (e.g., intent detector has new meta)
    # Assuming default detector is used after reset without overrides
    assert isinstance(engine._intent_detector, IntentDetectionInterface)
    assert engine._intent_detector is not initial_intent_detector  # Ensure new instance
    assert (
        engine._intent_detector._command_metadata == new_metadata
    )  # Check if new metadata was passed
    # train() needs to be called again after reset
    assert not engine._is_trained


# TODO: Add tests for default implementation logic once specified.
# TODO: Add tests for error handling (e.g., invalid overrides).
