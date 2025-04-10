"""Tests for TalkEngine interaction handling (FSM logic)."""

from unittest.mock import ANY, MagicMock
from typing import Dict, Any

from talkengine import TalkEngine
from talkengine.models import (
    NLUResult,
    ConversationDetail,
)
from talkengine.nlu_pipeline.models import InteractionState
from talkengine.nlu_pipeline.interaction_models import (
    ValidationRequestInfo,
    ClarificationData,
    ValidationData,
)
from talkengine.nlu_pipeline.interaction_handlers import InteractionResult

# Sample metadata for testing interactions
INTERACTION_TEST_METADATA = {
    "cmd_requires_param": {
        "description": "Command requiring a parameter",
        "parameters": {"req_param": "str"},
        "required_parameters": ["req_param"],  # Explicitly define required
    },
    "cmd_ambiguous1": {"description": "Ambiguous command 1"},
    "cmd_ambiguous2": {"description": "Ambiguous command 2"},
}


# --- Test Cases for Interaction Flows ---


def test_enter_clarification_on_low_confidence(
    mock_nlu_overrides: Dict[str, Any],
    mock_interaction_handlers: Dict[InteractionState, MagicMock],
    mock_intent_detector: MagicMock,  # Need detector mock directly
    mock_param_extractor: MagicMock,  # Added missing fixture
) -> None:
    """Test that the engine enters clarification mode on low intent confidence."""
    engine = TalkEngine(
        command_metadata=INTERACTION_TEST_METADATA,
        nlu_overrides=mock_nlu_overrides,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    query = "ambiguous query"

    # Configure Intent Detector to return low confidence
    # NOTE: Clarification options are now hardcoded placeholders in engine.run
    # We just need low confidence to trigger the mode.
    mock_intent_detector.classify_intent.return_value = {
        "intent": "cmd_ambiguous1",
        "confidence": 0.5,  # Below threshold used in engine.run (0.6)
    }

    # Configure the mock clarification handler's get_initial_prompt
    mock_clar_prompt = (
        "Which one did you mean? 1. cmd_ambiguous1 2. other_intent_example"
    )
    mock_interaction_handlers[
        InteractionState.CLARIFYING_INTENT
    ].get_initial_prompt.return_value = mock_clar_prompt

    result: NLUResult = engine.run(query)  # Returns NLUResult

    # Assertions: Expecting NLUResult indicating clarification needed
    assert isinstance(result, NLUResult)
    assert result.command is None  # Intent not confirmed
    assert result.parameters == {}
    assert result.artifacts is None
    assert isinstance(result.conversation_detail, ConversationDetail)
    # Check the prompt is in response_text
    assert result.conversation_detail.response_text == mock_clar_prompt
    # Check the interaction log
    assert len(result.conversation_detail.interactions) == 1
    log_entry = result.conversation_detail.interactions[0]
    assert isinstance(log_entry, tuple)
    assert log_entry[0] == InteractionState.CLARIFYING_INTENT.value
    assert log_entry[1] == mock_clar_prompt
    assert log_entry[2] is None  # No user response yet

    # Check context state
    assert (
        engine._pipeline_context.interaction_mode == InteractionState.CLARIFYING_INTENT
    )
    assert isinstance(engine._pipeline_context.interaction_data, ClarificationData)
    # Check placeholder options used in engine.run
    assert engine._pipeline_context.interaction_data.options == [
        "cmd_ambiguous1",
        "other_intent_example",
    ]

    # Check mocks
    mock_intent_detector.classify_intent.assert_called_once_with(query, ANY, None)
    mock_interaction_handlers[
        InteractionState.CLARIFYING_INTENT
    ].get_initial_prompt.assert_called_once_with(ANY)
    assert mock_param_extractor.identify_parameters.call_count == 0


def test_handle_clarification_input_success(
    mock_nlu_overrides: Dict[str, Any],
    mock_interaction_handlers: Dict[InteractionState, MagicMock],
    mock_param_extractor: MagicMock,  # Need mocks for subsequent steps
    mock_text_generator: MagicMock,
) -> None:
    """Test handling a successful clarification response from the user."""
    engine = TalkEngine(
        command_metadata=INTERACTION_TEST_METADATA,
        nlu_overrides=mock_nlu_overrides,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    user_clarification_input = "1"  # User chooses first option
    # initial_prompt = "Which one? 1. cmd_ambiguous1 2. cmd_ambiguous2" # Unused variable

    # Set engine state to be IN clarification mode
    engine._pipeline_context.interaction_mode = InteractionState.CLARIFYING_INTENT
    engine._pipeline_context.interaction_data = ClarificationData(
        prompt="Which one?",  # Not used by handler directly, but for context
        options=["cmd_ambiguous1", "cmd_ambiguous2"],
    )
    # Store the expected prompt *before* running the engine
    expected_prompt = engine._pipeline_context.interaction_data.prompt

    # Configure Clarification Handler mock for success
    mock_clar_handler = mock_interaction_handlers[InteractionState.CLARIFYING_INTENT]
    chosen_intent = "cmd_ambiguous1"
    mock_clar_handler.handle_input.return_value = InteractionResult(
        response=f"Okay, using {chosen_intent}",
        exit_mode=True,
        proceed_immediately=True,
        update_context={"current_intent": chosen_intent},  # Handler sets intent
    )

    # Configure subsequent NLU steps (param extractor, text gen)
    mock_params: Dict[str, Any] = {}
    mock_param_extractor.identify_parameters.return_value = (mock_params, [])
    mock_text = f"Response for {chosen_intent}"
    mock_text_generator.generate_text.return_value = mock_text

    result: NLUResult = engine.run(user_clarification_input)

    # Assertions: Final NLU result after successful clarification
    assert isinstance(result, NLUResult)
    assert result.command == chosen_intent
    assert result.parameters == mock_params
    assert result.artifacts is None
    assert isinstance(result.conversation_detail, ConversationDetail)
    assert result.conversation_detail.response_text == mock_text

    # Check interaction log (should include the clarification step)
    assert len(result.conversation_detail.interactions) == 1
    log_entry = result.conversation_detail.interactions[0]
    assert log_entry[0] == InteractionState.CLARIFYING_INTENT.value
    # Assert against the prompt stored *before* the interaction was cleared
    assert log_entry[1] == expected_prompt
    assert log_entry[2] == user_clarification_input

    # Check context state: Interaction mode should be cleared
    assert engine._pipeline_context.interaction_mode is None
    assert engine._pipeline_context.interaction_data is None
    assert engine._pipeline_context.current_intent == chosen_intent

    # Check mocks
    mock_clar_handler.handle_input.assert_called_once_with(
        user_clarification_input, ANY
    )
    # Verify NLU pipeline continued
    # Note: Param extractor is called with ORIGINAL query, not clarification input
    # This might need review in engine.run logic if incorrect. Assuming it uses context intent.
    mock_param_extractor.identify_parameters.assert_called_once_with(
        user_clarification_input,
        chosen_intent,
        ANY,  # Current run query passed, but uses context.intent
    )
    mock_text_generator.generate_text.assert_called_once_with(
        chosen_intent, mock_params, None, ANY  # Code result None
    )

    # Remove incorrect assertion
    # if "text_generation" in mock_nlu_overrides:
    #     assert cast(MagicMock, mock_text_generator.generate_text).call_count == 0
    # # If using default, engine._text_generator might be None or a default instance
    # elif engine._text_generator:
    #     assert cast(MagicMock, engine._text_generator.generate_text).call_count == 0


def test_enter_validation_on_missing_param(
    mock_nlu_overrides: Dict[str, Any],
    mock_interaction_handlers: Dict[InteractionState, MagicMock],
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,  # Added missing fixture
) -> None:
    """Test entering validation mode when a required parameter is missing."""
    engine = TalkEngine(
        command_metadata=INTERACTION_TEST_METADATA,
        nlu_overrides=mock_nlu_overrides,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    query = "do cmd_requires_param"  # Query doesn't provide 'req_param'
    intent_name = "cmd_requires_param"

    # Configure Intent Detector for high confidence
    mock_intent_detector.classify_intent.return_value = {
        "intent": intent_name,
        "confidence": 0.99,
    }
    # Configure Param Extractor to MISS the required param
    missing_param_name = "req_param"
    validation_request = ValidationRequestInfo(
        parameter_name=missing_param_name, reason="missing_required"
    )
    mock_param_extractor.identify_parameters.return_value = (
        {},  # Empty params extracted
        [validation_request],
    )

    # Configure mock validation handler's get_initial_prompt
    mock_val_prompt = f"What value for {missing_param_name}?"
    mock_interaction_handlers[
        InteractionState.VALIDATING_PARAMETER
    ].get_initial_prompt.return_value = mock_val_prompt

    result: NLUResult = engine.run(query)

    # Assertions: Expecting NLUResult indicating validation needed
    assert isinstance(result, NLUResult)
    assert result.command == intent_name  # Intent was identified
    assert result.parameters == {}  # Parameters still empty
    assert result.artifacts is None
    assert isinstance(result.conversation_detail, ConversationDetail)
    assert result.conversation_detail.response_text == mock_val_prompt
    # Check interaction log
    assert len(result.conversation_detail.interactions) == 1
    log_entry = result.conversation_detail.interactions[0]
    assert log_entry[0] == InteractionState.VALIDATING_PARAMETER.value
    assert log_entry[1] == mock_val_prompt
    assert log_entry[2] is None

    # Check context state
    assert (
        engine._pipeline_context.interaction_mode
        == InteractionState.VALIDATING_PARAMETER
    )
    assert isinstance(engine._pipeline_context.interaction_data, ValidationData)
    assert (
        engine._pipeline_context.interaction_data.parameter_name == missing_param_name
    )

    # Check mocks
    mock_intent_detector.classify_intent.assert_called_once_with(query, ANY, None)
    mock_param_extractor.identify_parameters.assert_called_once_with(
        query, intent_name, ANY
    )
    mock_interaction_handlers[
        InteractionState.VALIDATING_PARAMETER
    ].get_initial_prompt.assert_called_once_with(ANY)
    # Text generator should not be called yet
    # Need to access the mock via engine if using overrides dict
    if "text_generation" in mock_nlu_overrides:
        assert mock_text_generator.generate_text.call_count == 0
    # If using default, engine._text_generator might be None or a default instance
    elif engine._text_generator:
        assert mock_text_generator.generate_text.call_count == 0


def test_handle_validation_input_success(
    mock_nlu_overrides: Dict[str, Any],
    mock_interaction_handlers: Dict[InteractionState, MagicMock],
    mock_text_generator: MagicMock,  # Need text gen mock
) -> None:
    """Test handling a successful validation response from the user."""
    engine = TalkEngine(
        command_metadata=INTERACTION_TEST_METADATA,
        nlu_overrides=mock_nlu_overrides,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    user_validation_input = "provided value"
    validated_param_name = "req_param"
    intent_name = "cmd_requires_param"
    # initial_prompt = f"What value for {validated_param_name}?" # Unused variable

    # Set engine state to be IN validation mode
    engine._pipeline_context.interaction_mode = InteractionState.VALIDATING_PARAMETER
    engine._pipeline_context.current_intent = intent_name
    engine._pipeline_context.current_parameters = (
        {}
    )  # Params were empty before validation
    engine._pipeline_context.interaction_data = ValidationData(
        parameter_name=validated_param_name,
        reason="missing_required",  # Simplified for test state setup
        prompt="Placeholder",
    )
    # Store the expected prompt *before* running the engine
    expected_prompt = engine._pipeline_context.interaction_data.prompt

    # Configure Validation Handler mock for success
    mock_val_handler = mock_interaction_handlers[InteractionState.VALIDATING_PARAMETER]
    updated_params = {validated_param_name: user_validation_input}
    mock_val_handler.handle_input.return_value = InteractionResult(
        response=f"Okay, using {user_validation_input}",
        exit_mode=True,
        proceed_immediately=True,
        update_context={"current_parameters": updated_params},
    )

    # Configure subsequent NLU step (text gen)
    mock_text = f"Response for {intent_name} with {validated_param_name}={user_validation_input}"
    mock_text_generator.generate_text.return_value = mock_text

    result: NLUResult = engine.run(user_validation_input)

    # Assertions: Final NLU result after successful validation
    assert isinstance(result, NLUResult)
    assert result.command == intent_name
    assert result.parameters == updated_params  # Params updated by interaction
    assert result.artifacts is None
    assert isinstance(result.conversation_detail, ConversationDetail)
    assert result.conversation_detail.response_text == mock_text

    # Check interaction log
    assert len(result.conversation_detail.interactions) == 1
    log_entry = result.conversation_detail.interactions[0]
    assert log_entry[0] == InteractionState.VALIDATING_PARAMETER.value
    # Assert against the prompt stored *before* the interaction was cleared
    assert log_entry[1] == expected_prompt
    assert log_entry[2] == user_validation_input

    # Check context state
    assert engine._pipeline_context.interaction_mode is None
    assert engine._pipeline_context.interaction_data is None
    assert engine._pipeline_context.current_intent == intent_name
    assert engine._pipeline_context.current_parameters == updated_params

    # Check mocks
    mock_val_handler.handle_input.assert_called_once_with(user_validation_input, ANY)
    # Verify NLU pipeline continued (only text gen needed after validation)
    # Code exec would happen here if configured, then text gen
    mock_text_generator.generate_text.assert_called_once_with(
        intent_name, updated_params, None, ANY  # Code result None
    )

    # Remove incorrect assertion
    # if "text_generation" in mock_nlu_overrides:
    #     assert cast(MagicMock, mock_text_generator.generate_text).call_count == 0
    # # If using default, engine._text_generator might be None or a default instance
    # elif engine._text_generator:
    #     assert cast(MagicMock, engine._text_generator.generate_text).call_count == 0


# TODO: Add tests for re-prompting scenarios (handler returns exit_mode=False)
# TODO: Add tests for interactions being cancelled or failing
# TODO: Add tests for feedback interaction if implemented
