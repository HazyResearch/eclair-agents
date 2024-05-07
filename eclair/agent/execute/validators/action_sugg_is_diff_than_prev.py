from eclair.utils.logging import Action, Suggestion, TaskLog, Validation
from eclair.utils.executors import BaseValidator

class ActionSuggestionIsDiffThanPrevActionValidator(BaseValidator):
    """
    If an action is identical to the previous 2 actions and NOT a SCROLL (i.e. identical actuation), return INVALID.
    """

    def run(self, task_log: TaskLog) -> Validation:
        curr_action_suggestion: Suggestion = task_log.get_current_suggestion("action_suggestion")
        if curr_action_suggestion is None:
            return Validation(name='ActionSuggestionIsDiffThanPrevActionValidator', 
                              is_valid=True, 
                              feedback="No action suggestions have been made yet")
        if len(task_log.actions) < 2:
            return Validation(name='ActionSuggestionIsDiffThanPrevActionValidator', 
                              is_valid=True, 
                              feedback="There aren't 2 previous actions to compare to")

        # Ignore SCROLLs
        if "SCROLL" in curr_action_suggestion.action.upper():
            return Validation(name='ActionSuggestionIsDiffThanPrevActionValidator', 
                              is_valid=True, 
                              feedback="Current action is a SCROLL, which could be valid to repeat")
        
        # Get previous action
        prev_action: Action = task_log.actions[-1]
        prev_prev_action: Action = task_log.actions[-2]

        # If actions are identical, then return invalid
        if (curr_action_suggestion.action == prev_action.action and curr_action_suggestion.action == prev_prev_action.action):
            return Validation(name='ActionSuggestionIsDiffThanPrevActionValidator', 
                              is_valid=False, 
                              feedback="Current suggested action is identical to the most recent two actions. Are you sure you want to repeat this action? You might be stuck in a loop, or have chosen an action that does nothing given the current state of the webpage.")
        else:
            return Validation(name='ActionSuggestionIsDiffThanPrevActionValidator', 
                              is_valid=True, 
                              feedback="Current action is different than previous action")