from typing import Dict, Tuple
from eclair.utils.helpers import extract_coords_from_dsl_CLICK
from eclair.utils.logging import Action, State, Suggestion, TaskLog, Validation
from eclair.utils.executors import BaseValidator

class ClickCoordinatesWithinElementValidator(BaseValidator):
    """
    If a CLICK action's (x,y) coordinates are not contained in any element's bounding box, return INVALID.
    """

    def run(self, task_log: TaskLog) -> Validation:
        curr_action_suggestion: Suggestion = task_log.get_current_suggestion("action_suggestion")
        if curr_action_suggestion is None:
            return Validation(name='ClickCoordinatesWithinElementValidator', 
                              is_valid=True, 
                              feedback="No action suggestions have been made yet")

        # Ignore non-CLICKS
        if "CLICK" not in curr_action_suggestion.action.upper():
            return Validation(name='ClickCoordinatesWithinElementValidator', 
                              is_valid=True, 
                              feedback="Current action is not a CLICK.")

        # Click coords
        coords: Tuple[float, float] = extract_coords_from_dsl_CLICK(curr_action_suggestion.action)

        # Get current JSON state
        state: State = task_log.get_current_state()
        json_state: Dict[str, str] = state.json_state

        # Check if the coordinates are within any element's bounding box (x, y, width, height)
        for element in json_state:
            if (coords[0] >= element['x'] and coords[0] <= element['x'] + element['width'] and
                coords[1] >= element['y'] and coords[1] <= element['y'] + element['height']):
                return Validation(name='ClickCoordinatesWithinElementValidator', 
                                  is_valid=True, 
                                  feedback=f"Current CLICK action's coordinates of ({coords[0]}, {coords[1]}) are within the bounding box of element with xpath: {element['xpath']} (x={element['x']}, y={element['y']}, width={element['width']}, height={element['height']})")
        return Validation(name='ClickCoordinatesWithinElementValidator', 
                            is_valid=False, 
                            feedback=f"The CLICK coordinates of ({coords[0]}, {coords[1]}) are not within any element's bounding box on this webpage. Please try again, and make sure to specify valid coordinates for existing elements.")