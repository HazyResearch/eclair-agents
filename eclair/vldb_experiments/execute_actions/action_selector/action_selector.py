from typing import Any, Dict, List, Optional
from eclair.utils.helpers import (
    fetch_openai_text_completion,
    fetch_openai_vision_completion,
    encode_image,
)
from eclair.utils.logging import TaskLog, Suggestion
from eclair.utils.executors import BaseClass
from .prompts import PROMPTS


class ActionSelectorL1(BaseClass):
    """
    Purpose: Given a task, past actions, and current state, suggest the next action to take
    """

    def __init__(
        self,
        model_kwargs: Dict[str, str],
        task_outline: str = None,
    ):
        super().__init__(model_kwargs=model_kwargs)
        self.next_action_prompt_fn = PROMPTS["get_next_step_vision"]
        self.task_outline = task_outline
        print(self.task_outline)

    def run(
        self,
        previous_actions: List[str],
        is_vision: bool = False,
        json_state: Dict[str, str] = None,
        task: str = None,
        screen_shot: str = None,
    ) -> str:
        """Given a task, past actions, and current state, suggest the next action to take"""
        # Parse TaskLog
        task: str = task
        json_state: Dict[str, str] = json_state
        screen_shot: str = screen_shot
        task_outline: str = self.task_outline
        prior_feedback: Optional[str] = None
        prior_suggestion: Optional[str] = None

        # concatenate past_actions as a numerical list
        past_actions: str = "\n".join(
            [f"{str(i + 1)}" + ". " + x for i, x in enumerate(previous_actions)]
        )

        prompt: str = self.next_action_prompt_fn(
            task=task,
            json_state=json_state,
            task_outline=task_outline,
            past_actions=past_actions,
            prior_feedback=prior_feedback,
            prior_action=prior_suggestion,
        )

        print(prompt)

        # Get response
        if is_vision:
            response: str = fetch_openai_vision_completion(
                prompt,
                [screen_shot],
                temperature=0.0,  # model="gpt-4-vision-preview"
            )
            print(response)
        else:
            response: str = fetch_openai_text_completion(
                prompt,
                **self.model_kwargs,
            )

        return response
