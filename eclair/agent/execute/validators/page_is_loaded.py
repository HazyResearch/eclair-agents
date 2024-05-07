from typing import Dict, List, Tuple
from eclair.utils.helpers import encode_image
from eclair.utils.logging import State, TaskLog, Validation
from eclair.utils.helpers import fetch_openai_vision_completion
from eclair.utils.executors import BaseValidator

class PageIsLoadedValidator(BaseValidator):
    """
    If the page is still loading, then return INVALID.
    """
    def run(self, task_log: TaskLog) -> Validation:
        curr_state: State = task_log.get_current_state()
        curr_screenshot_base64: str = encode_image(curr_state.path_to_screenshot)
        
        # Query GPT-4V to see if page is still loading
        resp: str = fetch_openai_vision_completion("Here is a screenshot of a webpage. Please tell me if the page is still loading. Think step-by-step. If the page is still loading, end your response with the word 'The answer is YES'. Otherwise, end your response with the word 'The answer is NO'.", 
                                                   curr_screenshot_base64)
        
        is_loading: bool = False
        if "The answer is YES" in resp:
            is_loading = True
        elif "The answer is NO" in resp:
            is_loading = False
        elif "YES" in resp[-20:]:
            is_loading = True
        elif "NO" in resp[-20:]:
            is_loading = False
        elif "yes" in resp[-20:].lower():
            is_loading = True
        
        if is_loading:
            return Validation(
                name='PageIsLoadedValidator',
                is_valid=False, 
                feedback="The page has not finished loading yet."
            )
        else:
            return Validation(
                name='PageIsLoadedValidator',
                is_valid=True,
                feedback="The page has finished loading.",
            )