from typing import Dict, List, Optional, Tuple
from eclair.utils.logging import TaskLog, Validation
from eclair.utils.executors import BaseValidator

class FieldValueAfterTypeValidator(BaseValidator):
    """
    If a TYPE action has been executed, then require that the value of the element that was interacted with (if one exists)
    is equivalent to the argument `text` of that `TYPE(text)` call, otherwise return INVALID.
    """

    def run(self, task_log: TaskLog) -> Validation:
        curr_action: str = task_log.get_current_action().action
        if "TYPE" not in curr_action:
            return Validation(name='FieldValueAfterTypeValidator', 
                              is_valid=True, 
                              feedback="Current action is not a TYPE")
        
        # Get `value` attr of most recent interacted with element
        elem_attrs: Dict[str, str] = task_log.get_current_action().element_attributes
        value: Optional[str] = elem_attrs.get('element', {}).get('value', None)
        expected_value: str = curr_action.split("TYPE(")[-1].replace(")", "")
        
        if value != expected_value:
            return Validation(name='FieldValueAfterTypeValidator', 
                              is_valid=False, 
                              feedback="Value of element interacted with (`{value}`) does not match what was TYPE'd (`{expected_value}`)")
        else:
            return Validation(name='FieldValueAfterTypeValidator', 
                              is_valid=True, 
                              feedback="Value of element interacted with (`{value}`) matches what was TYPE'd (`{expected_value}`)")    
    