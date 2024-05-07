from typing import Dict, List, Optional, Tuple
from eclair.utils.logging import State, Suggestion, TaskLog, Validation
from eclair.utils.executors import BaseValidator

class FieldFocusedBeforeTypeValidator(BaseValidator):
    """
    If a TYPE action is requested, then require that a text field has been focused beforehand, otherwise return INVALID.
    """

    def run(self, task_log: TaskLog) -> Validation:
        curr_action_suggestion: Suggestion = task_log.get_current_suggestion("action_suggestion")
        if curr_action_suggestion is None:
            return Validation(name='FieldFocusedBeforeTypeValidator',
                              is_valid=True, 
                              feedback="No action suggestions have been made yet")

        if "TYPE" not in curr_action_suggestion.action.upper():
            return Validation(name='FieldFocusedBeforeTypeValidator',
                              is_valid=True, 
                              feedback="Current action is not a TYPE")
        
        curr_state: State = task_log.get_current_state()
        json_state: List[Dict[str, str]] = curr_state.json_state
        
        # Get currently focused element (if exists)
        focused_element: Optional[Dict[str, str]] = None
        for element in json_state:
            if element.get('is_focused'):
                focused_element = element
                break
        
        if focused_element is None:
            # No element is currently focused
            return Validation(name='FieldFocusedBeforeTypeValidator',
                              is_valid=False, 
                              feedback="No element is currently focused, but a TYPE action is requested. Please focus on a text field first.")
        else:
            # Element is currently focused - verify that it is a text field
            if focused_element.get('tag_name') == 'input' or focused_element.get('type') == 'text':
                return Validation(name='FieldFocusedBeforeTypeValidator',
                                  is_valid=True, 
                                  feedback="A text field element is currently focused")
            else:
                return Validation(name='FieldFocusedBeforeTypeValidator',
                                  is_valid=False, 
                                  feedback="The currently focused element is not a text field, but a TYPE action is requested. Please focus on a text field first.")