section__sop: str = lambda sop: f"""# Step-by-Step Guide

Here are the sequence of steps that were supposed to be followed to complete this workflow:
{sop}
"""

prompt__validate_task_completion__intro: str = lambda task_descrip, sop: f"""# Task
Your job is to decide whether the workflow was successfully completed, as depicted by the following sequence of screenshots.

# Workflow

The workflow is: "{task_descrip if task_descrip else 'Unknown'}"

# User Interface

The workflow was executed within the web application shown in the screenshots.

{section__sop(sop) if sop is not None else ''}

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
    "thinking": <think step by step what the answer should be>,
    "was_completed": <true/false>
}}

Please write your JSON below:
"""