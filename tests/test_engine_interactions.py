"""Tests for TalkEngine interaction handling (FSM logic)."""

from unittest.mock import ANY, MagicMock
from typing import Any
from copy import deepcopy

from talkengine import TalkEngine
from talkengine.models import (
    NLUResult,
    ConversationDetail,
    InteractionLogEntry,
)
from talkengine.nlu_pipeline.models import InteractionState, NLUPipelineState
from talkengine.nlu_pipeline.interaction_models import (
    ValidationRequestInfo,
    ClarificationData,
    ValidationData,
)

# from talkengine.nlu_pipeline.config import NLUOverridesConfig, CommandMetadataConfig # REMOVED incorrect import

# Import types directly
from talkengine.types import CommandMetadataConfig, NLUOverridesConfig  # Correct path

# Sample metadata for testing interactions
# Remove this if valid_command_metadata fixture is sufficient
# INTERACTION_TEST_METADATA = {
#     "cmd_requires_param": {
#         "description": "Command requiring a parameter",
#         "parameters": {"req_param": "str"},
#         "required_parameters": ["req_param"],  # Explicitly define required
#     },
#     "cmd_ambiguous1": {"description": "Ambiguous command 1"},
#     "cmd_ambiguous2": {"description": "Ambiguous command 2"},
# }


# --- Test Cases for Interaction Flows ---


def test_enter_clarification_on_low_confidence(
    valid_nlu_overrides_components_only: NLUOverridesConfig,
    mock_interaction_handlers: dict[InteractionState, MagicMock],
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    valid_command_metadata: CommandMetadataConfig,
) -> None:
    """Test that the engine enters clarification mode on low intent confidence."""
    engine = TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    query = "ambiguous query"
    intent_to_return = "test_command"

    # Configure Intent Detector to return low confidence
    mock_intent_detector.classify_intent.return_value = {
        "intent": intent_to_return,
        "confidence": 0.5,  # Below threshold used in engine.run (0.6)
    }

    # Configure the mock clarification handler's get_initial_prompt
    mock_clar_prompt = "Which one did you mean?"
    mock_interaction_handlers[
        InteractionState.CLARIFYING_INTENT
    ].get_initial_prompt.return_value = mock_clar_prompt

    result: NLUResult = engine.run(query)

    # Assertions: Expecting NLUResult indicating clarification needed
    assert isinstance(result, NLUResult)
    assert result.command is None
    assert result.parameters == {}
    assert result.artifacts is None
    assert isinstance(result.conversation_detail, ConversationDetail)
    assert result.conversation_detail.response_text == mock_clar_prompt
    assert len(result.conversation_detail.interactions) == 1
    log_entry = result.conversation_detail.interactions[0]
    assert isinstance(log_entry, InteractionLogEntry)
    assert log_entry.stage == InteractionState.CLARIFYING_INTENT.value
    assert log_entry.prompt == mock_clar_prompt
    assert log_entry.response is None

    # Check context state
    assert (
        engine._pipeline_context.interaction_mode == InteractionState.CLARIFYING_INTENT
    )
    assert isinstance(engine._pipeline_context.interaction_data, ClarificationData)
    assert engine._pipeline_context.interaction_data.options == [
        intent_to_return,
    ]

    # Check mocks
    mock_intent_detector.classify_intent.assert_called_once_with(
        user_input=query, context=ANY, excluded_intents=None
    )
    mock_interaction_handlers[
        InteractionState.CLARIFYING_INTENT
    ].get_initial_prompt.assert_called_once_with(ANY)
    assert mock_param_extractor.identify_parameters.call_count == 0


def test_handle_clarification_input_success(
    valid_nlu_overrides_components_only: NLUOverridesConfig,
    mock_interaction_handlers: dict[InteractionState, MagicMock],
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
    valid_command_metadata: CommandMetadataConfig,
) -> None:
    """Test handling a successful clarification response from the user."""
    engine = TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    user_clarification_input = "1"

    # Set engine state to be IN clarification mode
    chosen_intent = "test_command"
    other_option = "other_command"
    engine._pipeline_context.interaction_mode = InteractionState.CLARIFYING_INTENT
    engine._pipeline_context.interaction_data = ClarificationData(
        prompt="Which one?",
        options=[chosen_intent, other_option],
    )
    expected_prompt = engine._pipeline_context.interaction_data.prompt

    # Configure Clarification Handler mock for success
    mock_clar_handler = mock_interaction_handlers[InteractionState.CLARIFYING_INTENT]
    mock_final_context = deepcopy(engine._pipeline_context)
    mock_final_context.interaction_mode = None
    mock_final_context.interaction_data = None
    mock_final_context.current_intent = chosen_intent
    mock_final_context.confidence_score = 1.0

    mock_clar_handler.handle_input.return_value = (
        mock_final_context,
        True,
        NLUPipelineState.PARAMETER_IDENTIFICATION,
        None,
    )

    # Configure subsequent NLU steps (param extractor, text gen)
    mock_params: dict[str, Any] = {"required_param": 1.0}
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

    # Check interaction log
    assert len(result.conversation_detail.interactions) == 1
    log_entry = result.conversation_detail.interactions[0]
    assert log_entry[0] == InteractionState.CLARIFYING_INTENT.value
    assert log_entry[1] == expected_prompt
    assert log_entry[2] == user_clarification_input

    # Check context state
    assert engine._pipeline_context.interaction_mode is None
    assert engine._pipeline_context.interaction_data is None
    assert engine._pipeline_context.current_intent == chosen_intent

    # Check mocks
    mock_clar_handler.handle_input.assert_called_once_with(
        ANY, user_clarification_input
    )
    mock_param_extractor.identify_parameters.assert_called_once_with(
        user_input=user_clarification_input,
        intent=chosen_intent,
        parameter_class=valid_command_metadata[chosen_intent]["parameter_class"],
        context=ANY,
    )
    mock_text_generator.generate_text.assert_called_once_with(
        command=chosen_intent, parameters=mock_params, artifacts=None, context=ANY
    )


def test_enter_validation_on_missing_param(
    valid_nlu_overrides_components_only: NLUOverridesConfig,
    mock_interaction_handlers: dict[InteractionState, MagicMock],
    mock_intent_detector: MagicMock,
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
    valid_command_metadata: CommandMetadataConfig,
) -> None:
    """Test entering validation mode when a required parameter is missing."""
    engine = TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    intent_name = "test_command"
    query = f"do {intent_name}"

    # Configure Intent Detector for high confidence
    mock_intent_detector.classify_intent.return_value = {
        "intent": intent_name,
        "confidence": 0.99,
    }
    # Configure Param Extractor to MISS the required param
    missing_param_name = "required_param"
    validation_request = ValidationRequestInfo(
        parameter_name=missing_param_name, reason="missing_required"
    )
    mock_param_extractor.identify_parameters.return_value = (
        {"param1": "value"},
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
    assert result.command == intent_name
    assert result.parameters == {"param1": "value"}
    assert result.artifacts is None
    assert isinstance(result.conversation_detail, ConversationDetail)
    assert result.conversation_detail.response_text == mock_val_prompt
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
    # Check the requests list within ValidationData
    assert len(engine._pipeline_context.interaction_data.requests) == 1
    request_info = engine._pipeline_context.interaction_data.requests[0]
    assert isinstance(request_info, ValidationRequestInfo)
    assert request_info.parameter_name == missing_param_name
    assert request_info.reason == "missing_required"
    # Check prompt was set in interaction data
    assert engine._pipeline_context.interaction_data.prompt == mock_val_prompt

    # Check mocks
    mock_intent_detector.classify_intent.assert_called_once_with(
        user_input=query, context=ANY, excluded_intents=None
    )
    mock_param_extractor.identify_parameters.assert_called_once_with(
        user_input=query,
        intent=intent_name,
        parameter_class=valid_command_metadata[intent_name]["parameter_class"],
        context=ANY,
    )
    mock_interaction_handlers[
        InteractionState.VALIDATING_PARAMETER
    ].get_initial_prompt.assert_called_once_with(ANY)
    assert mock_text_generator.generate_text.call_count == 0


def test_handle_validation_input_success(
    valid_nlu_overrides_components_only: NLUOverridesConfig,
    mock_interaction_handlers: dict[InteractionState, MagicMock],
    mock_param_extractor: MagicMock,
    mock_text_generator: MagicMock,
    valid_command_metadata: CommandMetadataConfig,
) -> None:
    """Test handling a successful validation response from the user."""
    engine = TalkEngine(
        command_metadata=valid_command_metadata,
        nlu_overrides=valid_nlu_overrides_components_only,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    user_validation_input = "1.23"
    intent_name = "test_command"
    missing_param = "required_param"

    # Set engine state to be IN validation mode
    engine._pipeline_context.current_intent = intent_name
    engine._pipeline_context.current_parameters = {"param1": "abc"}
    engine._pipeline_context.interaction_mode = InteractionState.VALIDATING_PARAMETER
    engine._pipeline_context.interaction_data = ValidationData(
        requests=[
            ValidationRequestInfo(parameter_name=missing_param, reason="missing")
        ],
        prompt="What value?",
    )
    expected_prompt = engine._pipeline_context.interaction_data.prompt

    # Configure Validation Handler mock for success
    mock_val_handler = mock_interaction_handlers[InteractionState.VALIDATING_PARAMETER]
    updated_params = {"param1": "abc", missing_param: 1.23}

    mock_final_context = deepcopy(engine._pipeline_context)
    mock_final_context.interaction_mode = None
    mock_final_context.interaction_data = None
    mock_final_context.current_parameters = updated_params
    mock_final_context.parameter_validation_errors = []

    mock_val_handler.handle_input.return_value = (
        mock_final_context,
        True,
        NLUPipelineState.CODE_EXECUTION,
        None,
    )

    # Configure subsequent NLU steps (text gen - assume no code exec override here)
    mock_text = f"Response for {intent_name} with {updated_params}"
    mock_text_generator.generate_text.return_value = mock_text

    result: NLUResult = engine.run(user_validation_input)

    # Assertions: Final NLU result after successful validation
    assert isinstance(result, NLUResult)
    assert result.command == intent_name
    assert result.parameters == updated_params
    assert result.artifacts is None
    assert result.conversation_detail.response_text == mock_text

    # Check interaction log
    assert len(result.conversation_detail.interactions) == 1
    log_entry = result.conversation_detail.interactions[0]
    assert log_entry[0] == InteractionState.VALIDATING_PARAMETER.value
    assert log_entry[1] == expected_prompt
    assert log_entry[2] == user_validation_input

    # Check context state
    assert engine._pipeline_context.interaction_mode is None
    assert engine._pipeline_context.current_parameters == updated_params

    # Check mocks
    mock_val_handler.handle_input.assert_called_once_with(ANY, user_validation_input)
    assert mock_param_extractor.identify_parameters.call_count == 0
    mock_text_generator.generate_text.assert_called_once_with(
        command=intent_name, parameters=updated_params, artifacts=None, context=ANY
    )


# TODO: Add tests for re-prompting scenarios (handler returns exit_mode=False)
# TODO: Add tests for interactions being cancelled or failing
# TODO: Add tests for feedback interaction if implemented
