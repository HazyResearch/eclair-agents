from typing import Optional, List


def pre_filter_json_state(
    json_state: str, task: str, next_step: str, sop_step: str
) -> str:
    # format jsono state
    # split based on ",{"
    if json_state:
        json_state = [str(x) for x in json_state]
        json_state = "\n".join(json_state)

    # Previous Actions

    prompt = f"""# Task
    {task}

    # State

    We represent the current state of the webpage as a JSON list of elements of the form:

    {{
        'x' : x coordinate of element on screen (in pixels),
        'y' : y coordinate of element on screen (in pixels),
        'height' : height of element (in pixels),
        'width' : width of element (in pixels),
        'tag' : HTML tag of element,
        'text' : [optional] text contained within the element,
        'type' : [optional] type of element,
        'label' : [optional] label associated with element,
        'role' : [optional] accessibility role of element,
    }}

    The current state of the webpage is as follows:

    ```
    {json_state}
    ```

    # Current step of Standard Operating Procedure (SOP):
    {sop_step}

    # Next Step: 
    {next_step}

    # Given the next step, and the elements visible on the page
    
    (1) filter the elements to only those that are relevant to the next step. Format your response as a list of elements in the same format as the state above.

    (2) select the most relevent element. If there are multiple elements that correspond to the action, use the xpath and x,y coordinates (i.e., elements in the top right hand corner have the largest x value, and smallest y value) to select the element that is most appropriate. Explain your reasoning.
        
    # All Relevant elements:

    ```
    [
        {{"""
    return prompt


def get_next_action_vision_question_L1(
    task: str,
    json_state: str,
    past_actions: List[str],
    task_outline: Optional[str] = None,
    prior_feedback: Optional[str] = None,
    prior_action: Optional[str] = None,
    is_use_specific_action: bool = False,
    next_step: Optional[str] = None,
    sop_step: Optional[str] = None,
):
    # format jsono state
    # split based on ",{"
    if json_state:
        json_state = [str(x) for x in json_state]
        json_state = "\n".join(json_state)

    # Previous Actions
    if len(past_actions) == 0:
        prev_action_section: str = f"""None"""
    else:
        prev_action_section: str = f"""The previous actions taken were:

        {past_actions}"""

    prompt = f"""# Task
    {task}

    # State

    We represent the current state of the webpage as a JSON list of elements of the form:

    {{
        'x' : x coordinate of element on screen (in pixels),
        'y' : y coordinate of element on screen (in pixels),
        'height' : height of element (in pixels),
        'width' : width of element (in pixels),
        'tag' : HTML tag of element,
        'text' : [optional] text contained within the element,
        'type' : [optional] type of element,
        'label' : [optional] label associated with element,
        'role' : [optional] accessibility role of element,
    }}

    The current state of the webpage is as follows:

    ```
    {json_state}
    ```
    # Previous Actions

    {prev_action_section}

    # Current step of Standard Operating Procedure (SOP):

    {sop_step}

    # Next Step: {next_step}

    # Given the current state and the "Next Step", generate the next action in the format:
    {{ 
        "element": element to take action on (x,y coordinates, tag, label, name in dictionary form),
        "action": action sequence to take on element
    }}

    If there are multiple elements that correspond to the action, use the xpath and x,y coordinates to select the element that is most appropriate.

    Possible actions are:
    * `CLICK(x,y)`
        * where (x,y) are the coordinates on the screen of where to click
    * `TYPE(text)`
        * where text is the text that will be typed
    * `SCROLL(dy)`
        * where (dy) is the amount to scroll. Negative means scroll down, positive means scroll up. Scroll by multiples of 10.
    * `PRESS(key)`
        * where key is the key to press (e.g. "return", "up", "down", "left", "right")
    * `NAVIGATE(url)`
        * where url is the url to navigate to
    * `DELETE(x,y)`
        * where (x,y) are the coordinates of the text field to delete text from

    Example format:

    ```
    {{
        "element": {{'x': 100, 'y': 200, 'tag': 'button', 'text': 'To Date'}},
        "action": "CLICK(100, 200) | TYPE("01/01/2020")"
    }}
    ```
    # Notes:

    - For dropdown pickers: "action" = "CLICK | TYPE | PRESS('return')"
    - For calendars: "action" = "CLICK | TYPE | PRESS('return')"
    - If scrolling down, "action" = "SCROLL(-10)", "element" = "NONE"
    - For search bars: "action" = "CLICK | TYPE | PRESS('return')"
    - For resetting fields which contain text: "action" = "DELETE | CLICK | TYPE | PRESS('return')"
    - Only return a dictionary. No additional text.
    - Choose the most "related" element if not element is an exact match.

    Recall that the next step is: {next_step}

    # Given the current state of the UI, the SOP and your previous actions, what actions should you take:"""

    return prompt


def validate_pre_condition(
    pre_condition: str,
):
    prompt = f"""# Pre-condition
{pre_condition}

Is the pre-condition satisfied given the current state of the UI? Answer in the format:
{{
    "rationale": <rationale>,
    "is_satisfied": <true/false>
}}

Answer:"""
    return prompt


def validation_post_condition(
    post_condition: str,
):
    prompt = f"""# Post-condition
{post_condition}

Is the post-condition satisfied given the current state of the UI? Answer in the format:
{{
    "rationale": <rationale>,
    "is_satisfied": <true/false>
}}

Answer:"""


PROMPTS = {
    # "get_next_action_vision_question_L1": get_next_action_vision_question_L1,
    "convert_action_to_code": get_next_action_vision_question_L1,
    "pre_filter_json_state": pre_filter_json_state,
    "validate_pre_condition": validate_pre_condition,
    "validation_post_condition": validation_post_condition,
}
