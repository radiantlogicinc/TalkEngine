"""Tests for TalkEngine interaction handling (FSM logic)."""

from unittest.mock import ANY

from talkengine import TalkEngine
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
    mock_nlu_overrides, mock_interaction_handlers  # Use mock handlers
):
    """Test that the engine enters clarification mode on low intent confidence."""
    engine = TalkEngine(
        command_metadata=INTERACTION_TEST_METADATA,
        nlu_overrides=mock_nlu_overrides,
    )
    # Inject mock interaction handlers (conftest doesn't handle this automatically)
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    query = "ambiguous query"

    # Configure Intent Detector to return low confidence and options
    clarification_options = ["cmd_ambiguous1", "cmd_ambiguous2"]
    engine._intent_detector.classify_intent.return_value = {
        "intent": "cmd_ambiguous1",
        "confidence": 0.5,  # Below threshold (default 0.7)
        "options": clarification_options,
    }

    result, hint = engine.run(query)

    # Assertions
    assert hint == "awaiting_clarification"
    assert "interaction_prompt" in result
    assert result["interaction_prompt"] == "Mock Clarification Prompt"
    assert result["interaction_mode"] == InteractionState.CLARIFYING_INTENT

    # Check context state
    assert (
        engine._pipeline_context.interaction_mode == InteractionState.CLARIFYING_INTENT
    )
    assert isinstance(engine._pipeline_context.interaction_data, ClarificationData)
    assert engine._pipeline_context.interaction_data.options == clarification_options

    # Check mocks
    engine._intent_detector.classify_intent.assert_called_once_with(query, ANY)
    mock_interaction_handlers[
        InteractionState.CLARIFYING_INTENT
    ].get_initial_prompt.assert_called_once_with(ANY)
    engine._param_extractor.identify_parameters.assert_not_called()


def test_handle_clarification_input_success(
    mock_nlu_overrides, mock_interaction_handlers
):
    """Test handling a successful clarification response from the user."""
    engine = TalkEngine(
        command_metadata=INTERACTION_TEST_METADATA,
        nlu_overrides=mock_nlu_overrides,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    user_clarification_input = "1"  # User chooses first option

    # Set engine state to be IN clarification mode
    engine._pipeline_context.interaction_mode = InteractionState.CLARIFYING_INTENT
    engine._pipeline_context.interaction_data = ClarificationData(
        user_input="original query",
        options=["cmd_ambiguous1", "cmd_ambiguous2"],
        original_query="original query",
    )

    # Configure Clarification Handler mock for success
    mock_clar_handler = mock_interaction_handlers[InteractionState.CLARIFYING_INTENT]
    mock_clar_handler.handle_input.return_value = InteractionResult(
        response="Okay, using cmd_ambiguous1",
        exit_mode=True,
        proceed_immediately=True,
        update_context={
            "current_intent": "cmd_ambiguous1"
        },  # Handler sets the chosen intent
    )

    # Configure subsequent NLU steps (param extractor, text gen)
    # Assume cmd_ambiguous1 has no required params
    mock_params = {}
    mock_validation_reqs = []
    engine._param_extractor.identify_parameters.return_value = (
        mock_params,
        mock_validation_reqs,
    )
    mock_raw = {"raw": 1}
    mock_text = "Response for cmd_ambiguous1"
    engine._text_generator.generate_response.return_value = (mock_raw, mock_text)

    result, hint = engine.run(user_clarification_input)

    # Assertions: Final NLU result after successful clarification and pipeline continuation
    assert hint == "new_conversation"  # Final hint after successful run
    assert result["intent"] == "cmd_ambiguous1"  # Intent updated by interaction context
    assert result["parameters"] == mock_params
    assert result["raw_response"] == mock_raw
    assert result["response_text"] == mock_text
    assert "interaction_prompt" not in result

    # Check context state: Interaction mode should be cleared
    assert engine._pipeline_context.interaction_mode is None
    assert engine._pipeline_context.interaction_data is None
    assert engine._pipeline_context.current_intent == "cmd_ambiguous1"

    # Check mocks
    mock_clar_handler.handle_input.assert_called_once_with(
        user_clarification_input, ANY
    )
    # Verify NLU pipeline continued after proceed_immediately=True
    engine._param_extractor.identify_parameters.assert_called_once_with(
        user_clarification_input, "cmd_ambiguous1", ANY
    )
    engine._text_generator.generate_response.assert_called_once_with(
        "cmd_ambiguous1", mock_params, ANY
    )


def test_enter_validation_on_missing_param(
    mock_nlu_overrides, mock_interaction_handlers
):
    """Test entering validation mode when a required parameter is missing."""
    engine = TalkEngine(
        command_metadata=INTERACTION_TEST_METADATA,
        nlu_overrides=mock_nlu_overrides,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    query = "do cmd_requires_param"  # Query doesn't provide 'req_param'

    # Configure Intent Detector for high confidence
    engine._intent_detector.classify_intent.return_value = {
        "intent": "cmd_requires_param",
        "confidence": 0.99,
    }
    # Configure Param Extractor to MISS the required param
    missing_param_name = "req_param"
    engine._param_extractor.identify_parameters.return_value = (
        {},  # Empty params extracted
        [
            ValidationRequestInfo(
                parameter_name=missing_param_name, reason="missing_required"
            )
        ],
    )

    result, hint = engine.run(query)

    # Assertions
    assert hint == "awaiting_validation"
    assert "interaction_prompt" in result
    assert result["interaction_prompt"] == "Mock Validation Prompt"
    assert result["interaction_mode"] == InteractionState.VALIDATING_PARAMETER

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
    engine._intent_detector.classify_intent.assert_called_once_with(query, ANY)
    engine._param_extractor.identify_parameters.assert_called_once_with(
        query, "cmd_requires_param", ANY
    )
    mock_interaction_handlers[
        InteractionState.VALIDATING_PARAMETER
    ].get_initial_prompt.assert_called_once_with(ANY)
    engine._text_generator.generate_response.assert_not_called()


def test_handle_validation_input_success(mock_nlu_overrides, mock_interaction_handlers):
    """Test handling a successful validation response from the user."""
    engine = TalkEngine(
        command_metadata=INTERACTION_TEST_METADATA,
        nlu_overrides=mock_nlu_overrides,
    )
    engine._interaction_handlers = mock_interaction_handlers
    engine.train()
    user_validation_input = "provided value"  # User provides the missing param
    validated_param_name = "req_param"

    # Set engine state to be IN validation mode
    engine._pipeline_context.interaction_mode = InteractionState.VALIDATING_PARAMETER
    engine._pipeline_context.current_intent = (
        "cmd_requires_param"  # Need intent from previous step
    )
    engine._pipeline_context.interaction_data = ValidationData(
        user_input="original query",
        parameter_name=validated_param_name,
        error_message="Missing required parameter...",
    )

    # Configure Validation Handler mock for success
    mock_val_handler = mock_interaction_handlers[InteractionState.VALIDATING_PARAMETER]
    updated_params = {validated_param_name: user_validation_input}
    mock_val_handler.handle_input.return_value = InteractionResult(
        response=f"Okay, using {user_validation_input} for {validated_param_name}",
        exit_mode=True,
        proceed_immediately=True,
        update_context={
            "current_parameters": updated_params
        },  # Handler sets validated params
    )

    # Configure subsequent NLU step (text gen)
    mock_raw = {"raw": 2}
    mock_text = f"Response for cmd_requires_param with {validated_param_name}={user_validation_input}"
    engine._text_generator.generate_response.return_value = (mock_raw, mock_text)

    result, hint = engine.run(user_validation_input)

    # Assertions: Final NLU result after successful validation and pipeline continuation
    assert hint == "new_conversation"
    assert result["intent"] == "cmd_requires_param"
    assert (
        result["parameters"] == updated_params
    )  # Params updated by interaction context
    assert result["raw_response"] == mock_raw
    assert result["response_text"] == mock_text
    assert "interaction_prompt" not in result

    # Check context state: Interaction mode should be cleared
    assert engine._pipeline_context.interaction_mode is None
    assert engine._pipeline_context.interaction_data is None
    assert engine._pipeline_context.current_parameters == updated_params

    # Check mocks
    mock_val_handler.handle_input.assert_called_once_with(user_validation_input, ANY)
    # Verify NLU pipeline continued after proceed_immediately=True
    engine._param_extractor.identify_parameters.assert_not_called()  # Param extraction not re-run
    engine._text_generator.generate_response.assert_called_once_with(
        "cmd_requires_param", updated_params, ANY
    )


# TODO: Add tests for re-prompting scenarios (handler returns exit_mode=False)
# TODO: Add tests for interactions being cancelled or failing
# TODO: Add tests for feedback interaction if implemented
