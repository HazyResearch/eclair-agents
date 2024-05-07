from typing import Dict, List, Optional, Tuple
from eclair.utils.helpers import encode_image, is_diff_base64_images
from eclair.utils.logging import State, Suggestion, TaskLog, Validation
from eclair.utils.executors import BaseValidator

class ScrollChangedScreenValidator(BaseValidator):
    """
    If action is SCROLL and before/after screenshot of webpage is identical, then reached the end of scrollable element so INVALID
    """

    def run(self, task_log: TaskLog) -> Validation:
        curr_action_suggestion: Suggestion = task_log.get_current_suggestion("action_suggestion")
        if curr_action_suggestion is None:
            return Validation(name='ScrollChangedScreenValidator', 
                              is_valid=True, 
                              feedback="No action suggestions have been made yet")

        # Ignore if no before/after states available
        if len(task_log.states) < 2:
            return Validation(name='ScrollChangedScreenValidator', 
                              is_valid=True, 
                              feedback="Not enough states available to compare")

        # Ignore if not a SCROLL
        if "SCROLL" not in curr_action_suggestion.action.upper():
            return Validation(name='ScrollChangedScreenValidator', 
                              is_valid=True, 
                              feedback="Current action is not a SCROLL")

        # Get before/after states
        before_state: State = task_log.get_previous_state()
        after_state: State = task_log.get_current_state()
        
        # Load image for each state
        before_state_screenshot_base64: str = encode_image(before_state.path_to_screenshot)
        after_state_screenshot_base64: str = encode_image(after_state.path_to_screenshot)

        # Compare before/after screenshots
        if not is_diff_base64_images(before_state_screenshot_base64, after_state_screenshot_base64):
            return Validation(name='ScrollChangedScreenValidator', 
                              is_valid=False, 
                              feedback="The state of the webpage hasn't changed after scrolling, so we've probably reached the end of the scrollable element as there is no more content to view. Please choose a different action.")
        else:
            return Validation(name='ScrollChangedScreenValidator', 
                              is_valid=True, 
                              feedback="Before/after screenshots are different")