import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import dirtyjson
import openai
from eclair.utils.constants import SYSTEM_PROMPT
from eclair.utils.helpers import encode_image
from eclair.utils.logging import Action, TaskLog, State, Suggestion
from eclair.utils.executors import BaseClass
from .prompts import PROMPTS


class UniAgent(BaseClass):
    """
    Purpose: Feed GPT-4V a literal sequences of (S,A,S',A") in the form of screenshots followed by actions
    """

    def __init__(
        self,
        model_kwargs: Dict[str, str],
        intro_prompt: str,
        action_prompt: str,
        outro_prompt: str,
        sop: Optional[str] = None,
    ):
        super().__init__(model_kwargs=model_kwargs)
        self.intro_prompt_fn = PROMPTS[intro_prompt]
        self.action_prompt_fn = PROMPTS[action_prompt]
        self.outro_prompt_fn = PROMPTS[outro_prompt]
        self.sop = sop

    def next_action(self, task_log: TaskLog) -> Suggestion:
        client = openai.OpenAI()
        # Parse TaskLog
        task: str = task_log.task
        sop: str = self.sop
        env_type: str = task_log.env_type
        states: List[State] = task_log.states
        actions: List[Action] = task_log.get_actions()

        # Build (S,A) sequence where each S is a screenshot and each A is an action
        s_a_sequence: List[Union[State, Action]] = sorted(
            states + actions, key=lambda x: x.id, reverse=False
        )
        prompt_s_a_sequence: List[str] = []
        for item in s_a_sequence:
            if isinstance(item, State):
                if item.screenshot_base64 is None:
                    item.screenshot_base64 = encode_image(item.path_to_screenshot)
                assert (
                    item.screenshot_base64 is not None
                ), f"Error - Screenshot for state {item.id} is None"
                prompt_s_a_sequence.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{item.screenshot_base64}"
                                },
                            }
                        ],
                    }
                )
            elif isinstance(item, Action):
                prompt_s_a_sequence.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Given our overall task and history of screenshots and action, the next action we should take is: {item.action}",
                            }
                        ],
                    }
                )
            else:
                raise Exception(
                    f"Unknown type for `item` in `s_a_sequence`: {type(item)}"
                )

        # Task intro
        json_state: List[Dict[str, Any]] = task_log.get_current_state().json_state
        prompt_intro: Dict = {
            "role": "user",
            "content": [{"type": "text", "text": self.intro_prompt_fn(task)}],
        }
        prompt_first_action: Dict = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Please show me a screenshot of the webpage we are currently on.",
                }
            ],
        }

        # Feedback (if applicable)
        if (
            task_log.get_current_validation("action_suggestion")
            and not task_log.get_current_validation("action_suggestion").is_valid
        ):
            failed_prev_action: str = task_log.get_current_suggestion(
                "action_suggestion"
            ).action
            failed_prev_action_feedback: str = task_log.get_current_validation(
                "action_suggestion"
            ).feedback
        else:
            failed_prev_action = None
            failed_prev_action_feedback = None

        # Outro
        prompt_outro: Dict = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.outro_prompt_fn(
                        env_type,
                        task,
                        json_state,
                        sop,
                        actions,
                        failed_prev_action,
                        failed_prev_action_feedback,
                    ),
                }
            ],
        }

        # Construct full chat history
        messages: List[str] = (
            [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
            + [prompt_intro]
            + [prompt_first_action]
            + prompt_s_a_sequence
            + [prompt_outro]
        )
        assert len(messages) == (1 + 1 + 1 + len(states) + len(actions) + 1)

        # Send to OpenAI API
        try:
            response = client.chat.completions.create(
                messages=messages,
                max_tokens=4096,
                model="gpt-4-vision-preview",
                temperature=0,
            )
            response: str = response.choices[0].message.content
        except openai.RateLimitError:
            print("Rate limit exceeded -- waiting 1 min before retrying")
            time.sleep(60)
            return self.next_action(task_log)
        except openai.APIError as e:
            print(f"OpenAI API error: {e}")
            time.sleep(60)
            return self.next_action(task_log)
        except Exception as e:
            print(f"Unknown error: {e}")
            sys.exit(1)

        # Try to parse out JSON from response
        action_rationale: Optional[str] = None
        action: Optional[str] = None
        is_completed_answer: Optional[str] = None
        is_completed: bool = False
        try:
            action_resp = dirtyjson.loads(
                response[response.index("{") : response.rfind("}") + 1]
                .replace("```json", "")
                .replace("\n```", "")
                .replace("```", "")
                .strip("\n")
            )
            action_rationale = action_resp.get("action_rationale")
            action = action_resp.get("action")
            is_completed = action_resp.get("is_completed", False)
            is_completed_answer = action_resp.get("is_completed_answer")
        except Exception as e:
            print(str(e))
            print(f"Could not parse response: {response}")
            raise e

        # Cast blanks -> WAIT
        if action is None or action == "":
            action = "WAIT"

        return Suggestion(
            prompt=prompt_outro["content"][0]["text"],
            response=response,
            action=action,
            action_rationale=action_rationale,
            is_completed=is_completed,
            is_completed_answer=is_completed_answer,
        )
