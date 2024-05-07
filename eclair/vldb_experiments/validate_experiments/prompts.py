prompt__validate_condition: str = lambda condition: f"""# Task
You are an RPA bot that navigates digital UIs like a human. Your job is to validate that a certain condition is satisfied by the current state of the UI.

# Condition
{condition}

# Question

Given the current state of the UI shown in the screenshot, is the condition satisfied? 

Answer in the JSON format:
{{
    "rationale": <rationale>,
    "is_satisfied": <true/false>
}}

Answer:"""

prompt__validate_actuation: str = lambda next_action_suggestion, actuation: f"""# Task
You are an RPA bot that navigates digital UIs like a human. Your job is to validate that a certain action was successfully taken.

# Action
The action that was supposed to be taken was: "{next_action_suggestion}"

Specifically, the user did the following actions: `{actuation}`

# Question

The first screenshot shows the digital UI BEFORE the action was supposedly taken.
The second screenshot shows the digital UI AFTER the action was supposedly taken.

Given the change between the screenshots, was the action successfully taken? Be lenient and assume that the action was taken if the UI is "close enough" to the expected UI.

Answer in the JSON format:
{{
    "rationale": <rationale>,
    "was_taken": <true/false>
}}

Answer:"""
    
prompt__validate_task_completion__intro: str = lambda task_descrip: f"""# Task
Your job is to decide whether the workflow was successfully completed, as depicted by the following sequence of screenshots.

# Workflow

The workflow is: "{task_descrip}"

# User Interface

The workflow was executed within the web application shown in the screenshots.

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.

Between each screenshot, you are also provided the action that was taken to transition between screenshots. 

Here are the screenshots and actions of the workflow:"""

prompt__validate_task_completion__close: str = lambda : f"""
# Instructions

Given what you observe in the previous sequence of screenshots and actions, was the workflow successfully completed? 
If the workflow is asking a question, consider it completed successfully if you could deduce the answer to the question by viewing the screenshots. 
If the workflow was completed successfully, then set `was_completed` to `true`

Provide your answer as a JSON dictionary with the following format:
{{
    "rationale": <rationale>,
    "was_completed": <true/false>
}}

Please write your JSON below:
"""


prompt__validate_task_trajectory__intro: str = lambda task_descrip: f"""# Task
Your job is to decide whether the workflow that is demonstrated in the following sequence of screenshots ACCURATELY FOLLOWED the Step-by-Step Guide.

# Workflow

The workflow is: "{task_descrip}"

# User Interface

The workflow was executed within the web application shown in the screenshots.

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.

Between each screenshot, you are also provided the action that was taken to transition between screenshots. 

Here are the screenshots and actions of the workflow:"""

prompt__validate_task_trajectory__close: str = lambda sop : f"""

# Step-by-Step Guide

Here are the sequence of steps that you should have followed to complete this workflow:

{sop}

NOTE: The screenshots may not map 1-to-1 to the steps in the Step-by-Step Guide. i.e. screenshot #3 may correspond to step #2 (or multiple steps) in the Step-by-Step Guide.
However, as long as the general flow of the workflow is the same, then the workflow is considered to have accurately followed the Step-by-Step Guide.
Also note that elements may be interchangeably referred to as buttons or links (the distinction is not important).

# Instructions

Given what you observed in the previous sequence of screenshots and actions, was the Step-by-Step Guide accurately followed? If any of the steps are missing, or if any of the steps were performed out of order, then the Step-by-Step Guide was not accurately followed and `was_accurate` should be `false`.

Provide your answer as a JSON dictionary with the following format:
{{
    "rationale": <rationale>,
    "inaccurate_steps": <optional list of steps that were inaccurate>
    "was_accurate": <true/false>
}}

Please write your JSON below:
"""
