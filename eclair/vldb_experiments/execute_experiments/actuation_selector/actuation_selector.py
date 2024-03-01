from typing import Any, Dict, List, Optional
from eclair.utils.helpers import (
    fetch_openai_text_completion,
    fetch_openai_vision_completion,
    encode_image,
)
from eclair.utils.logging import TaskLog, Suggestion
from eclair.utils.executors import BaseClass
from .prompts import PROMPTS


class ActuationSelector(BaseClass):
    """
    Purpose: Given a task, past actions, and current state, suggest the next action to take
    L1: no task outline, no feedback, no selection of the indvidual "steps"
    """

    def __init__(
        self,
        model_kwargs: Dict[str, str],
    ):
        super().__init__(model_kwargs=model_kwargs)
        self.dsl_conv_prompt = PROMPTS["convert_action_to_code"]
        self.filter_json_prompt = PROMPTS["pre_filter_json_state"]
        self.pre_condition_prompt = PROMPTS["validate_pre_condition"]

    def run(
        self,
        previous_actions: List[str],
        action_suggestion: str,
        is_vision: bool = False,
        json_state: Dict[str, str] = None,
        task: str = None,
        screen_shot: str = None,
        sop_step: str = None,
    ) -> Suggestion:
        """Given a task, past actions, and current state, suggest the next action to take"""
        # Parse TaskLog
        task: str = task

        task_outline: str = None
        prior_feedback: Optional[str] = None
        prior_suggestion: Optional[str] = None

        # concatenate past_actions as a numerical list

        past_actions: str = "\n\n".join(
            [f"{str(i + 1)}" + ". " + x for i, x in enumerate(previous_actions)]
        )

        # pre-filter json state
        filtered_json_prompt: str = self.filter_json_prompt(
            json_state=json_state,
            task=task,
            next_step=action_suggestion,
            sop_step=sop_step,
        )

        # # Get response
        # if is_vision:
        filtered_json_state: str = fetch_openai_vision_completion(
            filtered_json_prompt,
            [screen_shot],
            temperature=0.0,  # model="gpt-4-vision-preview"
        )
        # else:
        # filtered_json_state: str = fetch_openai_text_completion(
        #     filtered_json_prompt,
        #     **self.model_kwargs,
        # )

        # Get actuation suggestion
        prompt: str = self.dsl_conv_prompt(
            task=task,
            json_state=str(filtered_json_state),
            task_outline=task_outline,
            past_actions=past_actions,
            prior_feedback=prior_feedback,
            prior_action=prior_suggestion,
            next_step=action_suggestion,
            sop_step=sop_step,
        )

        # Get response
        if is_vision:
            response: str = fetch_openai_vision_completion(
                prompt,
                [screen_shot],
                temperature=0.0,  # model="gpt-4-vision-preview"
            )
        else:
            response: str = fetch_openai_text_completion(
                prompt,
                **self.model_kwargs,
            )

        return filtered_json_state, response
