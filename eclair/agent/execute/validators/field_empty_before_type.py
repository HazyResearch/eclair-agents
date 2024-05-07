from typing import Dict, List, Optional, Tuple
from eclair.utils.logging import State, Suggestion, TaskLog, Validation
from eclair.utils.executors import BaseValidator

class FieldEmptyBeforeTypeValidator(BaseValidator):
    """
    If a TYPE action is requested, then require that the currently focused text field be currently empty, otherwise return INVALID.
    """

    def run(self, task_log: TaskLog) -> Validation:
        curr_action_suggestion: Suggestion = task_log.get_current_suggestion("action_suggestion")
        if curr_action_suggestion is None:
            return Validation(name='FieldEmptyBeforeTypeValidator',
                              is_valid=True, 
                              feedback="No action suggestions have been made yet")

        if "TYPE" not in curr_action_suggestion.action.upper():
            return Validation(name='FieldEmptyBeforeTypeValidator',
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
            return Validation(name='FieldEmptyBeforeTypeValidator',
                              is_valid=False, 
                              feedback="No element is currently focused")
        else:
            # Element is currently focused - verify that it is a text field
            if focused_element.get('tag_name') == 'input' or focused_element.get('type') == 'text':
                # Element is a text field - verify that it is currently empty
                if focused_element.get('value') in [None, '']:
                    return Validation(name='FieldEmptyBeforeTypeValidator',
                                      is_valid=True, 
                                      feedback="An empty text field element is currently focused")
                else:
                    print('---->', 'focused_element.value', focused_element.get('value'))
                    return Validation(name='FieldEmptyBeforeTypeValidator',
                                      is_valid=False, 
                                      feedback="The currently focused text field is not empty. You must clear it before typing.")
            else:
                return Validation(name='FieldEmptyBeforeTypeValidator',
                                  is_valid=False, 
                                  feedback="The currently focused element is not a text field.")