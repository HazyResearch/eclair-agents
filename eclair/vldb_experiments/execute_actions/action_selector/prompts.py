from typing import Optional, List


def select_step_vision(
    task: str,
    json_state: str,
    past_actions: List[str],
    task_outline: Optional[str] = None,
    prior_feedback: Optional[str] = None,
    prior_action: Optional[str] = None,
    is_use_specific_action: bool = False,
    next_step: Optional[str] = None,
):
    # json_state = [str(x) for x in json_state]
    # json_state = "\n".join(json_state)

    # Previous Actions
    if len(past_actions) == 0:
        prev_action_section: str = f"""None"""
    else:
        prev_action_section: str = f"""{past_actions}"""

    if task_outline is None:
        task_outline: str = ""
    else:
        task_outline: str = f"""# Steps of Procedure (SOP): \n{task_outline}"""

    current_step = len(past_actions.split("\n")) + 1
    last_action = past_actions.split("\n")[-1]

    prompt = f"""# Task
{task}

# Previous Actions
{prev_action_section}

{task_outline}

# Given the current state of the webpage, the SOP, and your previous actions, what is the next step to take from the SOP?

# Respond in the format: 
{{
    "previous action": <previous action>,
    "was the previous action successful": <true/false>,
    "next step": <next step>    
}}

# Next step:"""
    return prompt


def answer_question(
    task: str,
    json_state: str,
    past_actions: List[str],
    task_outline: Optional[str] = None,
    prior_feedback: Optional[str] = None,
    prior_action: Optional[str] = None,
    is_use_specific_action: bool = False,
    next_step: Optional[str] = None,
):
    prompt = f"""# Task
{task}

Given the contents of the current page, are you able to produce a final answer to the task?

# Respond in the format:
{{
    "rationale": <rationale>,
    "can_answer": <true/false>,
    "answer": <answer>,
    "confidence": <confidence>
}}

# Answer:"""
    return prompt


def get_next_step_vision(
    task: str,
    json_state: str,
    past_actions: List[str],
    task_outline: Optional[str] = None,
    prior_feedback: Optional[str] = None,
    prior_action: Optional[str] = None,
    is_use_specific_action: bool = False,
):
    # json_state = [str(x) for x in json_state]
    # json_state = "\n".join(json_state)

    # Previous Actions
    if len(past_actions) == 0:
        prev_action_section: str = f"""None"""
    else:
        prev_action_section: str = f"""{past_actions}"""

    if task_outline is None:
        next_instruction: str = (
            "Given the current state of the webpage, the task description, and your previous actions, suggest the next action in the format:"
        )
        rationale: str = (
            "single sentence rationale for action (the next step towards completing the task is...)"
        )
    else:
        next_instruction: str = (
            "Given the current state of the webpage, the SOP, and your previous actions, suggest the next action in the format:"
        )
        rationale: str = (
            'single sentence rationale for action (the next step in the "Steps of Procedure" we are on is...)'
        )

    if task_outline is None:
        task_outline: str = ""
    else:
        task_outline: str = f"""\n# Steps of Procedure (SOP): \n{task_outline}\n"""

    current_step = len(past_actions.split("\n")) + 1
    last_action = past_actions.split("\n")[-1]

    prompt = f"""# Task
{task}
{task_outline}
# Previous Actions
{prev_action_section}   

# {next_instruction}
{{
    "rationale": {rationale},
    "action": action sequence to take,
}}

# The action should be:
- a single step (i.e., click on a button, search for item in search bar, scroll down/up etc.). 
- before typing in an text field, input box or search bar, make sure you have "CLICK"ed on the field first (use the previous actions as a guide).
- Don't repeat actions unless the action was not completed before. 
- Scroll if you need to see more of the page.
- For calendars and dropdown pickers, type the value. 
- For dates adhere to the following format: "MM/DD/YYYY".
- If the task is complete, suggest "DONE".
- DO NOT EVER suggest a TYPE action (on a searchbar, input box or text field) that hasn't immediately been preceded by a CLICK action for that same element (use the previous actions to know what actions have been taken).

Given the current state of the UI and your previous actions, what actions should you take: CLICK, SCROLL, TYPE, PRESS etc.

# Next action:"""
    return prompt


def validate_pre_condition(
    pre_condition: str,
):
    prompt = f"""# Pre-condition
{pre_condition}

Is the pre-condition satisfied given the current state of the UI? Answer in the format:
{{
    "rationale": <rationale>,
    "is satisfied": <true/false>
}}

Answer:"""
    return prompt


PROMPTS = {
    "get_next_step_vision": get_next_step_vision,
    "select_step_vision": select_step_vision,
    "answer_question": answer_question,
    "validate_pre_condition": validate_pre_condition,
}

# - When interacting with a textfield or searchbar, make sure to first "CLICK on <field>" and then "TYPE <text>".
