"""Tests for the TalkEngine class."""

import pytest
from unittest.mock import MagicMock, ANY
from typing import Any
import logging  # Added for caplog config

from talkengine import TalkEngine
from talkengine.nlu_pipeline.nlu_engine_interfaces import (
    IntentDetectionInterface,
    ParameterExtractionInterface,
    TextGenerationInterface,
)
from talkengine.nlu_pipeline.models import (
    NLUPipelineContext,
)
from talkengine.models import (
    NLUResult,
    ConversationDetail,
)

# Import from Pydantic directly
from pydantic import BaseModel


# Import dummy models and types from conftest (assuming they are there)
from .conftest import DummyResult, dummy_exec_func_fail, dummy_exec_func_wrong_type

# Removed talkengine.types imports, use relative for now if needed or rely on fixture types
# from talkengine.types import NLUOverridesConfig, CommandMetadataConfig
# Import directly if needed for test function type hints
from talkengine.types import CommandMetadataConfig, NLUOverridesConfig


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


def test_talkengine_init_defaults(
    valid_command_metadata: CommandMetadataConfig,
) -> None:
    """Test initializing TalkEngine with default NLU components."""
    engine = TalkEngine(command_metadata=valid_command_metadata)
    assert engine._command_metadata == valid_command_metadata
    assert engine._conversation_history == []
    # Check the stored configuration dict
    assert engine._nlu_overrides_config == {}
    assert isinstance(engine._intent_detector, IntentDetectionInterface)
    assert isinstance(engine._param_extractor, ParameterExtractionInterface)
    assert isinstance(engine._text_generator, TextGenerationInterface)
    # Check that defaults were instantiated (or add specific checks if needed)
    assert not engine._is_trained
    assert isinstance(engine._pipeline_context, NLUPipelineContext)
    assert engine._pipeline_context.interaction_mode is None


def test_talkengine_init_with_history(
    valid_command_metadata: CommandMetadataConfig,
) -> None:
    """Test initializing TalkEngine with conversation history."""
    history = [{"role": "user", "content": "hi"}]
    engine = TalkEngine(
        command_metadata=valid_command_metadata, conversation_history=history
    )
    assert engine._conversation_history == history


def test_talkengine_init_with_overrides(
    valid_command_metadata: CommandMetadataConfig,
    valid_nlu_overrides_components_only: NLUOverridesConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test initializing TalkEngine with NLU overrides."""
    engine = TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
    assert engine._nlu_overrides_config == valid_nlu_overrides_components_only
    assert engine._intent_detector is mock_intent_detector
    assert engine._param_extractor is mock_param_extractor
    assert engine._text_generator is mock_text_generator
    assert isinstance(engine._pipeline_context, NLUPipelineContext)


def test_talkengine_train_placeholder(
    valid_command_metadata: CommandMetadataConfig,
) -> None:
    """Test the placeholder train() method."""
    engine = TalkEngine(command_metadata=valid_command_metadata)
    assert not engine._is_trained
    engine.train()  # Should run without error
    assert engine._is_trained


def test_talkengine_run_structure_defaults(
    valid_command_metadata: CommandMetadataConfig,
) -> None:
    """Test the structure of the run() method output with defaults (non-interactive path)."""
    engine = TalkEngine(command_metadata=valid_command_metadata)
    engine.train()
    # Use the simple command which has no required params
    result: NLUResult = engine.run("test query for simple_command")

    assert isinstance(result, NLUResult)
    # Default intent detector might find simple_command
    # Assuming default detector finds the intent based on description
    assert result.command == "simple_command"
    assert isinstance(result.parameters, dict)
    # Default param extractor now needs parameter_class, might extract or default
    # Should be empty for SimpleParams and simple query
    assert result.parameters == {}
    assert result.artifacts is None
    assert isinstance(result.conversation_detail, ConversationDetail)
    # Should be empty as SimpleParams has no required fields -> no validation interaction
    assert result.conversation_detail.interactions == []
    # Check default text gen output (adjust if default changes)
    assert (
        result.conversation_detail.response_text is not None
    )  # Check not None before 'in'
    assert "Intent: simple_command" in result.conversation_detail.response_text
    assert "Parameters: (no parameters)" in result.conversation_detail.response_text
    assert "Artifacts: None" in result.conversation_detail.response_text


def test_talkengine_run_with_overrides(
    valid_command_metadata: CommandMetadataConfig,
    valid_nlu_overrides_components_only: NLUOverridesConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test that run() calls the override methods (non-interactive path)."""
    engine = TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
    engine.train()
    query = "another test query"
    intent = "test_command"

    # Configure mocks for a successful non-interactive run
    mock_intent_detector.classify_intent.return_value = {
        "intent": intent,
        "confidence": 0.99,
    }
    mock_params = {"required_param": 1.0}
    mock_param_extractor.identify_parameters.return_value = (mock_params, [])
    mock_text = "mock_text"
    mock_text_generator.generate_text.return_value = mock_text

    result: NLUResult = engine.run(query)

    assert isinstance(result, NLUResult)
    assert result.command == intent
    assert result.parameters == mock_params
    assert result.artifacts is None
    assert isinstance(result.conversation_detail, ConversationDetail)
    assert result.conversation_detail.interactions == []
    assert result.conversation_detail.response_text == mock_text

    # Check if the correct mock instances were called with context and correct args
    mock_intent_detector.classify_intent.assert_called_once_with(
        user_input=query, context=ANY, excluded_intents=None
    )
    mock_param_extractor.identify_parameters.assert_called_once_with(
        user_input=query,
        intent=intent,
        parameter_class=valid_command_metadata[intent]["parameter_class"],
        context=ANY,
    )
    mock_text_generator.generate_text.assert_called_once_with(
        command=intent, parameters=mock_params, artifacts=None, context=ANY
    )


def test_talkengine_reset(valid_command_metadata: CommandMetadataConfig) -> None:
    """Test resetting the engine."""
    history1 = [{"role": "user", "content": "q1"}]
    engine = TalkEngine(
        command_metadata=valid_command_metadata, conversation_history=history1
    )
    engine.train()
    assert engine._command_metadata == valid_command_metadata
    assert engine._conversation_history == history1
    assert engine._is_trained
    initial_intent_detector = engine._intent_detector

    class NewParamsReset(BaseModel):
        pass

    # Add explicit type hint for CommandMetadataConfig
    new_metadata: CommandMetadataConfig = {
        "new_cmd": {"description": "New", "parameter_class": NewParamsReset}
    }
    new_history: list[dict[str, Any]] = []
    engine.reset(command_metadata=new_metadata, conversation_history=new_history)

    assert engine._command_metadata == new_metadata
    assert engine._conversation_history == new_history
    assert isinstance(engine._intent_detector, IntentDetectionInterface)
    assert engine._intent_detector is not initial_intent_detector
    assert isinstance(engine._pipeline_context, NLUPipelineContext)
    assert engine._pipeline_context.interaction_mode is None
    assert not engine._is_trained


# TODO: Add tests for default implementation logic once specified.
# TODO: Add tests for error handling (e.g., invalid overrides).

# --- New Tests for Refactored Engine ---


def test_talkengine_run_excluded_intents(
    valid_command_metadata: CommandMetadataConfig,
    valid_nlu_overrides_components_only: NLUOverridesConfig,
    mock_intent_detector: MagicMock,
) -> None:
    """Test run() correctly passes excluded_intents to intent detector."""
    engine = TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
    engine.train()
    query = "some query"
    exclude = ["other_command"]

    engine.run(query, excluded_intents=exclude)

    mock_intent_detector.classify_intent.assert_called_once_with(
        user_input=query, context=ANY, excluded_intents=exclude
    )


# --- Tests for Code Execution (Using New Structure) --- #


def test_talkengine_run_with_code_execution(
    valid_command_metadata: CommandMetadataConfig,
    valid_nlu_overrides_exec_code_only: NLUOverridesConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test run() correctly executes code specified in nlu_overrides."""
    overrides = valid_nlu_overrides_exec_code_only.copy()
    overrides["intent_detection"] = mock_intent_detector
    overrides["param_extraction"] = mock_param_extractor
    overrides["text_generation"] = mock_text_generator

    engine = TalkEngine(
        command_metadata=valid_command_metadata, nlu_overrides=overrides
    )
    engine.train()
    query = "run test command with required 1.23 and param1 abc"
    intent = "test_command"

    mock_intent_detector.classify_intent.return_value = {
        "intent": intent,
        "confidence": 0.99,
    }
    extracted_params = {"param1": "abc", "required_param": 1.23}
    mock_param_extractor.identify_parameters.return_value = (extracted_params, [])
    mock_text_generator.generate_text.return_value = "Code executed and text generated."

    result: NLUResult = engine.run(query)

    assert result.command == intent
    assert result.parameters == extracted_params
    assert isinstance(result.artifacts, DummyResult)
    assert result.artifacts.output == "Processed abc"
    assert result.artifacts.status_code == 200
    assert (
        result.conversation_detail.response_text == "Code executed and text generated."
    )

    mock_intent_detector.classify_intent.assert_called_once_with(
        user_input=query, context=ANY, excluded_intents=None
    )
    mock_param_extractor.identify_parameters.assert_called_once_with(
        user_input=query,
        intent=intent,
        parameter_class=valid_command_metadata[intent]["parameter_class"],
        context=ANY,
    )
    mock_text_generator.generate_text.assert_called_once_with(
        command=intent,
        parameters=extracted_params,
        artifacts=result.artifacts,
        context=ANY,
    )


def test_talkengine_run_only_text_generation(
    valid_command_metadata: CommandMetadataConfig,
    valid_nlu_overrides_components_only: NLUOverridesConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test run() when only text generation is configured (no code exec override)."""
    engine = TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
    engine.train()
    query = "run test command"
    intent = "test_command"

    mock_intent_detector.classify_intent.return_value = {
        "intent": intent,
        "confidence": 0.99,
    }
    mock_params = {"required_param": 1.0}
    mock_param_extractor.identify_parameters.return_value = (mock_params, [])
    mock_text_generator.generate_text.return_value = "Only text generated."

    result: NLUResult = engine.run(query)

    assert result.command == intent
    assert result.artifacts is None
    assert result.conversation_detail.response_text == "Only text generated."
    mock_text_generator.generate_text.assert_called_once_with(
        command=intent, parameters=mock_params, artifacts=None, context=ANY
    )


def test_talkengine_run_no_code_or_text(
    valid_command_metadata: CommandMetadataConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
) -> None:
    """Test run() with only default text generation (no code exec, no text override)."""
    overrides_no_text: NLUOverridesConfig = {
        "intent_detection": mock_intent_detector,
        "param_extraction": mock_param_extractor,
    }
    engine = TalkEngine(
        command_metadata=valid_command_metadata, nlu_overrides=overrides_no_text
    )
    engine.train()
    query = "run test command"
    intent = "test_command"

    mock_intent_detector.classify_intent.return_value = {
        "intent": intent,
        "confidence": 0.99,
    }
    mock_params = {"required_param": 1.0}
    mock_param_extractor.identify_parameters.return_value = (mock_params, [])

    expected_text_parts = [
        f"Intent: {intent}",
        f"Parameters: {mock_params}",
        "Artifacts: None",
    ]

    result: NLUResult = engine.run(query)

    assert result.command == intent
    assert result.artifacts is None
    # Assert that the default text generator produced output containing expected parts
    response = result.conversation_detail.response_text
    assert response is not None
    for part in expected_text_parts:
        assert part in response


# --- Initialization Tests (Using Fixtures from conftest) --- #


def test_talkengine_init_valid(valid_command_metadata, empty_nlu_overrides) -> None:
    """Test valid initialization."""
    engine = TalkEngine(valid_command_metadata, nlu_overrides=empty_nlu_overrides)
    assert engine._command_metadata == valid_command_metadata
    assert engine._nlu_overrides_config == empty_nlu_overrides
    assert isinstance(engine._intent_detector, IntentDetectionInterface)
    assert isinstance(engine._param_extractor, ParameterExtractionInterface)
    assert isinstance(engine._text_generator, TextGenerationInterface)


# Add tests for invalid metadata (using fixtures)
def test_talkengine_init_invalid_metadata_type() -> None:
    with pytest.raises(TypeError, match="command_metadata must be a dictionary"):
        TalkEngine(command_metadata=None)  # type: ignore


def test_talkengine_init_invalid_metadata_structure(
    invalid_command_metadata_missing_desc,
) -> None:
    with pytest.raises(ValueError, match="missing string 'description'"):
        TalkEngine(invalid_command_metadata_missing_desc)


def test_talkengine_init_invalid_metadata_param_class_type(
    invalid_command_metadata_param_not_basemodel,
) -> None:
    with pytest.raises(TypeError, match="must be a subclass of pydantic.BaseModel"):
        TalkEngine(invalid_command_metadata_param_not_basemodel)


# Add tests for invalid overrides (using fixtures)
def test_talkengine_init_invalid_overrides_type(valid_command_metadata) -> None:
    with pytest.raises(TypeError, match="nlu_overrides must be a dictionary"):
        TalkEngine(valid_command_metadata, nlu_overrides=123)  # type: ignore


def test_talkengine_init_invalid_overrides_nlu_component(
    valid_command_metadata,
) -> None:
    with pytest.raises(
        TypeError, match="Override for 'intent_detection' must implement"
    ):
        TalkEngine(valid_command_metadata, nlu_overrides={"intent_detection": object()})  # type: ignore [dict-item]


def test_talkengine_init_invalid_overrides_exec_structure(
    valid_command_metadata, invalid_nlu_overrides_bad_exec_structure
) -> None:
    with pytest.raises((ValueError, TypeError), match="executable_code"):
        TalkEngine(
            valid_command_metadata,
            nlu_overrides=invalid_nlu_overrides_bad_exec_structure,
        )


def test_talkengine_init_invalid_overrides_exec_func_type(
    valid_command_metadata, invalid_nlu_overrides_non_callable_func
) -> None:
    with pytest.raises(ValueError, match="must include a callable 'function'"):
        TalkEngine(
            valid_command_metadata,
            nlu_overrides=invalid_nlu_overrides_non_callable_func,
        )


def test_talkengine_init_invalid_overrides_exec_result_type(
    valid_command_metadata, invalid_nlu_overrides_result_not_basemodel
) -> None:
    with pytest.raises(TypeError, match="must be a subclass of pydantic.BaseModel"):
        TalkEngine(
            valid_command_metadata,
            nlu_overrides=invalid_nlu_overrides_result_not_basemodel,
        )


# --- Run Method Tests (Using Fixtures from conftest) --- #


def test_talkengine_run_with_component_overrides(
    valid_command_metadata: CommandMetadataConfig,
    valid_nlu_overrides_components_only: NLUOverridesConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test that run() calls the override NLU components."""
    engine = TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
    engine.train()
    query = "run test command"
    intent = "test_command"
    params = {"required_param": 1.0}

    mock_intent_detector.classify_intent.return_value = {
        "intent": intent,
        "confidence": 0.99,
    }
    mock_param_extractor.identify_parameters.return_value = (params, [])
    mock_text_generator.generate_text.return_value = "mock_text"

    result: NLUResult = engine.run(query)

    assert result.command == intent
    assert result.parameters == params
    assert result.artifacts is None
    assert result.conversation_detail.response_text == "mock_text"

    mock_param_extractor.identify_parameters.assert_called_once_with(
        user_input=query,
        intent=intent,
        parameter_class=valid_command_metadata[intent]["parameter_class"],
        context=ANY,
    )
    mock_text_generator.generate_text.assert_called_once_with(
        command=intent, parameters=params, artifacts=None, context=ANY
    )


def test_talkengine_run_exec_code_success(
    valid_command_metadata: CommandMetadataConfig,
    valid_nlu_overrides_exec_code_only: NLUOverridesConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test run() correctly executes code via nlu_overrides."""
    overrides = valid_nlu_overrides_exec_code_only.copy()
    overrides["intent_detection"] = mock_intent_detector
    overrides["param_extraction"] = mock_param_extractor
    overrides["text_generation"] = mock_text_generator

    engine = TalkEngine(
        command_metadata=valid_command_metadata, nlu_overrides=overrides
    )
    engine.train()
    query = "run test command with required 1.23 and param1 abc"

    mock_intent_detector.classify_intent.return_value = {
        "intent": "test_command",
        "confidence": 0.99,
    }
    extracted_params = {"param1": "abc", "required_param": 1.23}
    mock_param_extractor.identify_parameters.return_value = (extracted_params, [])
    mock_text_generator.generate_text.return_value = (
        "Dummy executed and text generated."
    )

    result: NLUResult = engine.run(query)

    assert result.command == "test_command"
    assert result.parameters == extracted_params
    assert isinstance(result.artifacts, DummyResult)
    assert result.artifacts.output == "Processed abc"
    assert result.artifacts.status_code == 200
    assert (
        result.conversation_detail.response_text == "Dummy executed and text generated."
    )

    mock_intent_detector.classify_intent.assert_called_once_with(
        user_input=query, context=ANY, excluded_intents=None
    )
    mock_param_extractor.identify_parameters.assert_called_once_with(
        user_input=query,
        intent="test_command",
        parameter_class=valid_command_metadata["test_command"]["parameter_class"],
        context=ANY,
    )
    mock_text_generator.generate_text.assert_called_once_with(
        command="test_command",
        parameters=extracted_params,
        artifacts=result.artifacts,
        context=ANY,
    )


def test_talkengine_run_no_exec_code_defined(
    valid_command_metadata: CommandMetadataConfig,
    valid_nlu_overrides_components_only: NLUOverridesConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
) -> None:
    """Test run() when no executable_code is defined for the intent."""
    engine = TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
    engine.train()
    query = "run test command"
    intent = "test_command"
    params = {"required_param": 1.0}

    mock_intent_detector.classify_intent.return_value = {
        "intent": intent,
        "confidence": 0.99,
    }
    mock_param_extractor.identify_parameters.return_value = (params, [])
    mock_text_generator.generate_text.return_value = "No code ran."

    result = engine.run(query)

    assert result.command == intent
    assert result.parameters == params
    assert result.artifacts is None
    assert result.conversation_detail.response_text == "No code ran."
    mock_text_generator.generate_text.assert_called_once_with(
        command=intent, parameters=params, artifacts=None, context=ANY
    )


def test_talkengine_run_exec_code_param_validation_error(
    valid_command_metadata: CommandMetadataConfig,
    valid_nlu_overrides_exec_code_only: NLUOverridesConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
    caplog,  # Add caplog fixture
) -> None:
    """Test run() handles Pydantic ValidationError during parameter instantiation."""
    # Temporarily enable propagation for the specific logger
    fw_logger = logging.getLogger("fastWorkflow")
    original_propagate = fw_logger.propagate
    fw_logger.propagate = True
    # caplog doesn't need specific targeting now
    # caplog.set_level(logging.ERROR, logger="fastWorkflow")

    try:
        overrides = valid_nlu_overrides_exec_code_only.copy()
        overrides["intent_detection"] = mock_intent_detector
        overrides["param_extraction"] = mock_param_extractor
        overrides["text_generation"] = mock_text_generator

        engine = TalkEngine(
            command_metadata=valid_command_metadata, nlu_overrides=overrides
        )
        engine.train()
        query = "run test command"
        intent = "test_command"
        extracted_params = {"param1": "abc"}

        mock_intent_detector.classify_intent.return_value = {
            "intent": intent,
            "confidence": 0.99,
        }
        mock_param_extractor.identify_parameters.return_value = (extracted_params, [])
        mock_text_generator.generate_text.return_value = "Text after validation error"

        result = engine.run(query)

        assert result.command == intent
        assert result.parameters == extracted_params
        assert result.artifacts is None
        assert (
            "Pydantic validation error instantiating DummyParams for 'test_command'"
            in caplog.text
        )
        mock_text_generator.generate_text.assert_called_once_with(
            command=intent, parameters=extracted_params, artifacts=None, context=ANY
        )
        assert result.conversation_detail.response_text == "Text after validation error"
    finally:
        # Restore original propagation setting
        fw_logger.propagate = original_propagate


def test_talkengine_run_exec_code_function_exception(
    valid_command_metadata: CommandMetadataConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
    caplog,  # Add caplog fixture
) -> None:
    """Test run() handles exceptions raised by the executable function."""
    # Temporarily enable propagation for the specific logger
    fw_logger = logging.getLogger("fastWorkflow")
    original_propagate = fw_logger.propagate
    fw_logger.propagate = True
    # caplog doesn't need specific targeting now
    # caplog.set_level(logging.ERROR, logger="fastWorkflow")

    try:
        # Use override with the failing function
        overrides: NLUOverridesConfig = {
            "intent_detection": mock_intent_detector,
            "param_extraction": mock_param_extractor,
            "text_generation": mock_text_generator,
            "test_command": {
                "executable_code": {
                    "function": dummy_exec_func_fail,
                    "result_class": DummyResult,
                }
            },
        }
        engine = TalkEngine(
            command_metadata=valid_command_metadata, nlu_overrides=overrides
        )
        engine.train()
        query = "run test command"
        intent = "test_command"
        extracted_params = {"param1": "abc", "required_param": 1.23}

        mock_intent_detector.classify_intent.return_value = {
            "intent": intent,
            "confidence": 0.99,
        }
        mock_param_extractor.identify_parameters.return_value = (extracted_params, [])
        mock_text_generator.generate_text.return_value = "Text after function error"

        result = engine.run(query)

        assert result.command == intent
        assert result.parameters == extracted_params
        assert result.artifacts is None
        assert (
            "Error executing function for command 'test_command': Dummy execution failed"
            in caplog.text
        )
        assert "Dummy execution failed" in caplog.text
        mock_text_generator.generate_text.assert_called_once_with(
            command=intent, parameters=extracted_params, artifacts=None, context=ANY
        )
        assert result.conversation_detail.response_text == "Text after function error"
    finally:
        # Restore original propagation setting
        fw_logger.propagate = original_propagate


def test_talkengine_run_exec_code_wrong_return_type(
    valid_command_metadata: CommandMetadataConfig,
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
    caplog,  # Add caplog fixture
) -> None:
    """Test run() handles executable function returning the wrong type."""
    # Temporarily enable propagation for the specific logger
    fw_logger = logging.getLogger("fastWorkflow")
    original_propagate = fw_logger.propagate
    fw_logger.propagate = True
    # caplog doesn't need specific targeting now
    # caplog.set_level(logging.ERROR, logger="fastWorkflow")

    try:
        overrides: NLUOverridesConfig = {  # Correct type
            "intent_detection": mock_intent_detector,
            "param_extraction": mock_param_extractor,
            "text_generation": mock_text_generator,
            "test_command": {
                "executable_code": {
                    "function": dummy_exec_func_wrong_type,
                    "result_class": DummyResult,
                }
            },
        }
        engine = TalkEngine(
            command_metadata=valid_command_metadata, nlu_overrides=overrides
        )
        engine.train()
        query = "run test command"
        intent = "test_command"
        extracted_params = {"param1": "abc", "required_param": 1.23}

        mock_intent_detector.classify_intent.return_value = {
            "intent": intent,
            "confidence": 0.99,
        }
        mock_param_extractor.identify_parameters.return_value = (extracted_params, [])
        mock_text_generator.generate_text.return_value = "Text after type error"

        result = engine.run(query)

        assert result.command == intent
        assert result.parameters == extracted_params
        assert result.artifacts is None
        assert (
            "Executable function for 'test_command' returned type dict, expected DummyResult"
            in caplog.text
        )
        mock_text_generator.generate_text.assert_called_once_with(
            command=intent, parameters=extracted_params, artifacts=None, context=ANY
        )
        assert result.conversation_detail.response_text == "Text after type error"
    finally:
        # Restore original propagation setting
        fw_logger.propagate = original_propagate
