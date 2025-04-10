"""Handlers for different NLU pipeline interaction modes (clarification, validation, feedback)."""

# New file
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Import the specific data models
from .interaction_models import (
    ClarificationData,
    FeedbackData,
    ValidationData,
)
from .models import NLUPipelineContext
from ..utils.logging import logger


@dataclass
class InteractionResult:
    """Structured result from an interaction handler."""

    response: str  # Message for the user
    exit_mode: bool = False  # Should the manager exit this interaction mode?
    proceed_immediately: bool = (
        False  # Should the manager re-run core logic after exiting?
    )
    update_context: Optional[Dict[str, Any]] = field(
        default_factory=dict
    )  # NLUPipelineContext fields to update
    error_message: Optional[str] = None  # For reporting input processing errors


class BaseInteractionHandler(ABC):
    """Abstract base class for interaction handlers."""

    @abstractmethod
    def get_initial_prompt(self, context: NLUPipelineContext) -> str:
        """Generate the initial prompt for this interaction mode."""

    @abstractmethod
    def handle_input(
        self, user_input: str, context: NLUPipelineContext
    ) -> InteractionResult:
        """Process user input during this interaction mode."""


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
        self, user_input: str, context: NLUPipelineContext
    ) -> InteractionResult:
        """Processes user's choice during clarification."""
        logger.debug(f"ClarificationHandler handling input: '{user_input}'")

        if not isinstance(context.interaction_data, ClarificationData):
            logger.error(
                "ClarificationHandler: Invalid interaction data type on input."
            )
            # Exit interaction, proceed with default/unknown state?
            return InteractionResult(
                exit_mode=True,
                proceed_immediately=False,
                response="Error: Invalid interaction data for clarification.",
            )

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

        if chosen_intent:
            # Update context directly (this part is okay)
            context.current_intent = chosen_intent
            context.confidence_score = 1.0  # Assume high confidence after clarification
            # Exit clarification, proceed to parameter extraction
            return InteractionResult(
                exit_mode=True,
                proceed_immediately=True,
                response=f"Okay, proceeding with {chosen_intent}.",
            )
        else:
            # Clarification failed, maybe reprompt or exit?
            # For now, exit and let the engine handle the 'unknown' intent state
            return InteractionResult(
                exit_mode=True,
                proceed_immediately=False,  # Don't proceed if clarification failed
                response="Sorry, I didn't understand that choice. Please try again.",
            )


class ValidationHandler(BaseInteractionHandler):
    """Handles the parameter validation interaction."""

    def get_initial_prompt(self, context: NLUPipelineContext) -> str:
        """Generates the validation prompt."""
        if not isinstance(context.interaction_data, ValidationData):
            logger.error("ValidationHandler: Invalid interaction data type.")
            return "Sorry, I need more information. Could you please rephrase?"

        data: ValidationData = context.interaction_data
        prompt = (
            data.prompt
            or f"What is the value for {data.parameter_name}? ({data.reason})"
        )
        logger.debug(f"Generated validation prompt: {prompt}")

        return prompt

    def handle_input(
        self, user_input: str, context: NLUPipelineContext
    ) -> InteractionResult:
        """Processes user's input for a missing/invalid parameter."""
        logger.debug(f"ValidationHandler handling input: '{user_input}'")

        if not isinstance(context.interaction_data, ValidationData):
            logger.error("ValidationHandler: Invalid interaction data type on input.")
            return InteractionResult(
                exit_mode=True,
                proceed_immediately=False,
                response="Error: Invalid interaction data for validation.",
            )

        data: ValidationData = context.interaction_data
        parameter_name = data.parameter_name
        validated_value = user_input  # Assume the input is the value for now
        # TODO: Add type validation/conversion based on metadata?

        logger.info(f"User provided value for {parameter_name}: '{validated_value}'")

        # Update the specific parameter in the context
        # Ensure current_parameters is initialized if not already
        if context.current_parameters is None:
            context.current_parameters = {}
        context.current_parameters[parameter_name] = validated_value

        # Exit validation mode, proceed to code execution (or next step)
        return InteractionResult(
            exit_mode=True,
            proceed_immediately=True,
            response=f"Okay, using '{validated_value}' for {parameter_name}.",
        )


class FeedbackHandler(BaseInteractionHandler):
    """Handles user feedback on the response."""

    def get_initial_prompt(self, context: NLUPipelineContext) -> str:
        data = context.interaction_data
        if not isinstance(data, FeedbackData):
            logger.error("FeedbackHandler: Invalid interaction data type.")
            return "Could I get your feedback on the previous response?"  # Fallback

        # Maybe truncate long responses for the prompt
        response_snippet = (
            data.response_text[:200] + "..."
            if len(data.response_text) > 200
            else data.response_text
        )
        return f"Regarding the response:\n---\n{response_snippet}\n---\n{data.prompt}"

    def handle_input(
        self, user_message: str, context: NLUPipelineContext
    ) -> InteractionResult:
        # Remove logging access to non-existent attributes
        # if context.last_prompt_shown:
        #     log_entry: InteractionLogEntry = (
        #         context.interaction_mode.value if context.interaction_mode else "feedback", # Guard access
        #         context.last_prompt_shown,
        #         user_message,
        #     )
        # context.recorded_interactions.append(log_entry) # Remove access

        # Replace _get_typed_data with direct access and type check
        data = context.interaction_data
        if not isinstance(data, FeedbackData):
            logger.error("FeedbackHandler: Invalid interaction data type on input.")
            return InteractionResult(
                response="Error processing feedback.", exit_mode=True
            )

        # Placeholder Logic: Just acknowledge feedback and exit mode
        feedback = user_message.strip().lower()
        # Here you would log the feedback, potentially adjust future responses, etc.
        print(f"Received feedback: {feedback}")  # Replace with logging

        response_message = "Thanks for the feedback!"
        if feedback in ["no", "incorrect", "wrong"]:
            # Could trigger a re-generation attempt or ask for more details
            response_message = "Thanks for letting me know. Can you provide more details on what was wrong?"
            # Decide if we should exit mode or ask clarifying question here. For now, exit.

        return InteractionResult(
            response=response_message,
            exit_mode=True,
            proceed_immediately=False,  # Usually don't proceed automatically after feedback
        )


# Add other handlers as needed
