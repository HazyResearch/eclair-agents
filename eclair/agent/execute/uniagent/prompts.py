from typing import Any, Dict, List, Optional

from eclair.utils.logging import Action

def intro_prompt(task: str):
    prompt: str = f"""
You are a robotic process automation (RPA) agent that can perform any task on a computer. Your goal is to complete the task described below.
After you suggest an action, you will be shown a screenshot of the application, and you will suggest the next action to take.

# Task

{task}

"""
    return prompt

def action_prompt(action: str, step: int):
    prompt: str = f"""
### Action #{step}

{action}
"""
    return prompt

def outro_prompt(env_type: str,
                 task: str, 
                 json_state: List[Dict[str, Any]], 
                 sop: Optional[str], 
                 action_history: List[Action],
                 failed_prev_action: Optional[str], 
                 failed_prev_action_feedback: Optional[str]):
    is_env_browser: bool = env_type in ["selenium", "playwright", ]

    # Provide JSON State (it available)
    section__json_state = "NO JSON STATE PROVIDED."
    if json_state is not None:
        section__json_state = str(json_state).replace("\'", '"')

    # Explicitly pull out currently focused element (if applicable)
    section__focused_element: str = ""
    if is_env_browser:
        # We'll only have this information in browser-based environments
        if isinstance(json_state, list): # If we are using m2w, json_state is just html and not iterable
            # Highlight the currently focused element
            section__focused_element = f"IMPORTANT NOTE: No element is currently focused."
            for element in json_state:
                if element.get('is_focused'):
                    section__focused_element = f"IMPORTANT NOTE: The currently focused element is: `{element}`"
                    break

    # Provide action history
    section__action_history: str = ""
    if action_history:
        action_history = "\n".join([f"{idx + 1}. {action.action}: {action.rationale}" for idx, action in enumerate(action_history)])
        section__action_history = f"""
## Action History

Here is a history of the actions you have taken so far:

```
{action_history}
```
"""

    # Provide SOP (if applicable)
    section__sop: str = ""
    if sop:
        section__sop = f"""
## Standard Operating Procedure (SOP)

This section contains a list of steps that are typically taken to complete the task. It can help guide your action selection. Here is the SOP for this task:

```
{sop}
```
"""

    # Provide feedback from prev failed action (if applicable)
    section__failed_prev_action: str = ""
    if failed_prev_action:
        section__failed_prev_action += f"""
### Feedback from Previous Attempt

You previously attempted to suggest the action `{failed_prev_action}`. However, this was rejected as invalid. The reason for this rejection was: `{failed_prev_action_feedback}`.
"""

    # Construct prompt
    prompt: str = f"""

## Most Recent State

In addition to the above screenshot, we have pulled the relevant elements from the screen and represented them as a JSON list of elements of the form:

    {{
        "x" : x coordinate of element on screen (in pixels),
        "y" : y coordinate of element on screen (in pixels),
        "height" : height of element (in pixels),
        "width" : width of element (in pixels),
        "tag" : HTML tag of element,
        "text" : [optional] text contained within the element,
        "type" : [optional] type of element,
        "label" : [optional] label associated with element,
        "role" : [optional] accessibility role of element,
        {'"is_focused" : [optional] true if the element is currently focused,' if is_env_browser else ''}
        {'"is_disabled" : [optional] true if the element is currently disabled,' if is_env_browser else ''}
        {'"is_checked" : [optional] true if the element is currently checked,' if is_env_browser else ''}
    }}

The JSON representation of the application shown in the most recent screenshot is:

```
{section__json_state}
```


{section__focused_element}


{section__sop}


{section__action_history}


## Next Action Suggestion

Your job now is to output a JSON object with the following fields:

    {{
        "is_completed_rationale" : [string - required] that explains why you think the task is complete or not complete.
        "is_completed" : [boolean - required] that is TRUE if the task is complete and FALSE otherwise.
        "is_completed_answer" : [string - optional] If `is_completed=True`, provide the answer to the question specified in the task. Otherwise, leave this field blank.
        "action_rationale": [string - optional] If `is_completed=False`, then explain your rationale for why you are suggested that ACTION here. Otherwise, leave this field blank.
        "action" : [ACTION - optional] If `is_completed=False`, then provide the ACTION to take next here. Otherwise, leave this field blank.
        "action_expected_outcome": [string - optional] If `is_completed=False`, then suggest what we expect to happen to the page after taking the ACTION. Otherwise, leave this field blank.
    }}


Instructions for filling out the JSON object:

#### If Task is Complete...

1. Consider whether the task is complete. As a reminder, your task was: "{task}". Think step-by-step through what information you need to answer the question and if that information is present in the current state. 
2. If the task is complete, set `is_completed` to TRUE and, if the task asks a question, provide an answer to that question in the `is_completed_answer` field.

### If Task is Not Complete...

1. Set `is_completed` to FALSE and leave `is_completed_answer` blank.
2. Fill in `action` with the next immediate ACTION to take, where ACTION must be one of the following:

    * `CLICK(x,y)`
        * where (x,y) are the coordinates on the screen of where to click
    * `TYPE(text)`
        * where text is the text that will be typed
    * `SCROLL(dx,dy)`
        * where (dx,dy) is the amount to scroll in the x and y directions respectively
    * `PRESS(key)`
        * where key is the key to press (must be one of: "return", "up", "down", "left", "right")
    * `CLEAR()`
        * clears the currently focused element (i.e. if you have focused a text input, this will clear the text in that input)
    * `WAIT()`
        * waits for a few seconds before re-observing the screen state

3. Keep these rules in mind when suggesting an ACTION:

    1. If you are TYPING TEXT, you must have first FOCUSED the element that you want to type into.
    2. If the screen you are currently on is relevant to the task, but you do not see any relevant elements to interact with, then consider SCROLLING to reveal more of the application. However, if you have scrolled down multiple times in a row without finding the element you are looking for, you should stop scrolling and suggest a different action since you've probably reached the bottom of the page.
    3. ONLY act on elements THAT ARE INCLUDED in the current screen. In other words, they must be part of the current state listed above.
    4. For certain tasks, note that you might have to click buttons with names like "Save" or "Submit" (if these buttons are present) to complete the task and save your work. However, note that forms might autosave, so if you can't find these buttons don't get stuck looking for them.
    5. Only suggest ONE ACTION.
    6. If an SOP is provided, use the SOP to guide your next action.
    7. Take into account ALL of your previous actions.
    8. Your output must be valid JSON!


### Examples

Note that the following examples are just that, examples. They are not relevant to your task, but are provided to give you an idea of what the JSON object should look like.

If task is not completed:
{{
    "is_completed_rationle" : "The task has not been finished.",
    "is_completed" : false,
    "is_completed_answer" : "",
    "action_rationale" : "I clicked on the button with the text "Submit" because I thought that would submit the form."
    "action" : "CLICK(100,200)"
    "action_expected_outcome": "I expect the page to refresh and show a confirmation message."
}}

If task is completed:
{{
    "is_completed_rationle" : "We can answer the question by looking at the screenshot.",
    "is_completed" : true,
    "is_completed_answer" : "The answer to the question is 42.",
    "action_rationale" : "",
    "action" : ""
    "action_expected_outcome": ""
}}

The above examples are not relevant to your task, and are only provided to show you how to format your JSON output.

{section__failed_prev_action}


### Action Suggestion for this Task

DO NOT REPEAT the last action. Only output the JSON object -- nothing else. For this task, your action suggestion JSON object is:
"""
    return prompt


PROMPTS = {
    "intro_prompt": intro_prompt,
    "action_prompt": action_prompt,
    "outro_prompt": outro_prompt,
}
