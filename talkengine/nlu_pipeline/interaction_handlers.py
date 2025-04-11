"""Handlers for different NLU pipeline interaction modes (clarification, validation, feedback)."""

# New file
from abc import ABC, abstractmethod
from typing import Optional

# Import the specific data models
from .interaction_models import (
    ClarificationData,
    FeedbackData,
    ValidationData,
)
from .models import NLUPipelineContext, NLUPipelineState  # Import state enum
from ..utils.logging import logger


class BaseInteractionHandler(ABC):
    """Abstract base class for interaction handlers."""

    @abstractmethod
    def get_initial_prompt(self, context: NLUPipelineContext) -> str:
        """Generate the initial prompt for this interaction mode."""

    @abstractmethod
    def handle_input(
        self, context: NLUPipelineContext, user_input: str
    ) -> tuple[NLUPipelineContext, bool, Optional[str], Optional[str]]:
        """Process user input during this interaction mode.

        Returns:
            A tuple containing:
            - The updated NLUPipelineContext.
            - A boolean indicating if the main pipeline should proceed immediately.
            - An optional string indicating the next pipeline step (use NLUPipelineState values).
            - An optional string response to show the user IF proceed_immediately is False.
        """


class ClarificationHandler(BaseInteractionHandler):
    """Handles the intent clarification interaction."""

    def get_initial_prompt(self, context: NLUPipelineContext) -> str:
        """Generates the clarification prompt with options."""
        if not isinstance(context.interaction_data, ClarificationData):
            logger.error("ClarificationHandler: Invalid interaction data type.")
            return "Sorry, I got confused. Could you please rephrase?"

        data: ClarificationData = context.interaction_data
        options_text = "\n".join(
            f"{i+1}. {option}" for i, option in enumerate(data.options)
        )
        prompt = f"{data.prompt}\n{options_text}"
        logger.debug(f"Generated clarification prompt: {prompt}")

        return prompt

    def handle_input(
        self, context: NLUPipelineContext, user_input: str
    ) -> tuple[NLUPipelineContext, bool, Optional[str], Optional[str]]:
        """Processes user's choice during clarification."""
        logger.debug(f"ClarificationHandler handling input: '{user_input}'")

        if not isinstance(context.interaction_data, ClarificationData):
            logger.error(
                "ClarificationHandler: Invalid interaction data type on input."
            )
            # Exit interaction, don't proceed immediately, no next step
            context.interaction_mode = None
            context.interaction_data = None
            error_response = "Error: Invalid interaction data for clarification."
            return context, False, None, error_response

        data: ClarificationData = context.interaction_data
        chosen_intent: Optional[str] = None

        # Try to map input to an option index
        try:
            choice_index = int(user_input.strip()) - 1
            if 0 <= choice_index < len(data.options):
                chosen_intent = data.options[choice_index]
                logger.info(f"User clarified intent: {chosen_intent}")
            else:
                logger.warning("User input is not a valid option number.")
        except ValueError:
            # Maybe try fuzzy matching the text input against options?
            logger.warning("User input is not a number. Clarification failed.")

        context.interaction_mode = None  # Exit interaction mode
        context.interaction_data = None
        if chosen_intent:
            # Update context directly
            context.current_intent = chosen_intent
            context.confidence_score = 1.0  # Assume high confidence after clarification
            # Exit clarification, proceed immediately to parameter extraction
            return context, True, NLUPipelineState.PARAMETER_IDENTIFICATION.value, None
        else:
            fail_response = "Sorry, I didn't understand that choice. Please try again."
            # Don't proceed immediately, no specific next step (engine might retry intent classification or fail)
            return context, False, None, fail_response


class ValidationHandler(BaseInteractionHandler):
    """Handles the parameter validation interaction."""

    def get_initial_prompt(self, context: NLUPipelineContext) -> str:
        """Generates the validation prompt."""
        if not isinstance(context.interaction_data, ValidationData):
            logger.error("ValidationHandler: Invalid interaction data type.")
            return "Sorry, I need more information. Could you please rephrase?"

        data: ValidationData = context.interaction_data
        # Process the first request, assuming one for now
        if not data.requests:
            logger.error("ValidationHandler: No validation requests found in data.")
            return "Sorry, something went wrong with validation. Could you rephrase?"

        first_request = data.requests[0]
        prompt = (
            data.prompt  # Use prompt set by engine if available
            or f"What is the value for {first_request.parameter_name}? ({first_request.reason})"
        )
        logger.debug(f"Generated validation prompt: {prompt}")
        # The engine will set last_prompt_shown based on this return value
        return prompt

    def handle_input(
        self, context: NLUPipelineContext, user_input: str
    ) -> tuple[NLUPipelineContext, bool, Optional[str], Optional[str]]:
        """Processes user's input for a missing/invalid parameter."""
        logger.debug(f"ValidationHandler handling input: '{user_input}'")

        if not isinstance(context.interaction_data, ValidationData):
            logger.error("ValidationHandler: Invalid interaction data type on input.")
            context.interaction_mode = None
            context.interaction_data = None
            error_response = "Error: Invalid interaction data for validation."
            return context, False, None, error_response

        data: ValidationData = context.interaction_data
        # Process the first request, assuming one for now
        if not data.requests:
            logger.error(
                "ValidationHandler: No validation requests found in data during input."
            )
            context.interaction_mode = None
            context.interaction_data = None
            error_response = "Error: No validation request details found."
            return context, False, None, error_response

        first_request = data.requests[0]
        parameter_name = first_request.parameter_name  # Get name from request
        validated_value = user_input  # Assume the input is the value for now
        # TODO: Add type validation/conversion based on metadata?

        logger.info(f"User provided value for {parameter_name}: '{validated_value}'")

        # Update the specific parameter in the context
        # Ensure current_parameters is initialized if not already
        if context.current_parameters is None:
            context.current_parameters = {}
        context.current_parameters[parameter_name] = validated_value

        context.interaction_mode = None  # Exit validation mode
        context.interaction_data = None

        # Exit validation mode, proceed immediately to code execution (or next step)
        return context, True, NLUPipelineState.CODE_EXECUTION.value, None


class FeedbackHandler(BaseInteractionHandler):
    """Handles user feedback on the response."""

    def get_initial_prompt(self, context: NLUPipelineContext) -> str:
        data = context.interaction_data
        if not isinstance(data, FeedbackData):
            logger.error("FeedbackHandler: Invalid interaction data type.")
            return "Could I get your feedback on the previous response?"  # Fallback

        # Maybe truncate long responses for the prompt
        response_snippet = (
            f"{data.response_text[:200]}..."
            if len(data.response_text) > 200
            else data.response_text
        )
        prompt = f"Regarding the response:\n---\n{response_snippet}\n---\n{data.prompt}"
        # Engine sets last_prompt_shown
        return prompt

    def handle_input(
        self, context: NLUPipelineContext, user_message: str
    ) -> tuple[NLUPipelineContext, bool, Optional[str], Optional[str]]:
        # sourcery skip: extract-duplicate-method, inline-variable

        data = context.interaction_data
        if not isinstance(data, FeedbackData):
            logger.error("FeedbackHandler: Invalid interaction data type on input.")
            context.interaction_mode = None
            context.interaction_data = None
            error_response = "Error processing feedback."
            return context, False, None, error_response

        # Placeholder Logic: Just acknowledge feedback and exit mode
        feedback = user_message.strip().lower()
        # Here you would log the feedback, potentially adjust future responses, etc.
        print(f"Received feedback: {feedback}")  # Replace with logging

        response_message = "Thanks for the feedback!"
        if feedback in {"no", "incorrect", "wrong"}:
            # Could trigger a re-generation attempt or ask for more details
            response_message = "Thanks for letting me know. Can you provide more details on what was wrong?"
            # Decide if we should exit mode or ask clarifying question here. For now, exit.

        context.interaction_mode = None  # Exit feedback mode
        context.interaction_data = None

        # Usually don't proceed automatically after feedback
        return context, False, None, response_message


# Add other handlers as needed
