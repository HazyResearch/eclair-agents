import json
import subprocess
import time
import os
import traceback
import openai
import sys
import platform
import base64
import surya.postprocessing.text
from typing import Callable, Optional, List, Dict, Any, Tuple, Union
from collections import namedtuple
from dataclasses import dataclass
from playwright.sync_api import sync_playwright, Browser
from selenium import webdriver
from PIL import Image, ImageDraw, ImageFont
from eclair.utils.constants import SYSTEM_PROMPT
from eclair.utils.logging import TaskLog, Validation, ScreenRecorder, LIST_OF_BROWSER_APPLICATIONS
from moviepy.editor import VideoFileClip

# A mutable flag to indicate if the signal handler is currently running
is_handler_running_flag = [False]

def get_png_size(img_path):
    with Image.open(img_path) as img:
        width, height = img.size
    return width, height

def signal_handler(
    sig,
    frame,
    path_to_output_dir: str,
    trace_name: str,
    screen_recorder: ScreenRecorder,
    is_handler_running_flag: List[bool,],
) -> None:
    """Handles Ctrl+C kill process signal to save logs before exiting"""
    if is_handler_running_flag[0]:
        print("You pressed Ctrl+C twice -- Quitting immediately")
        sys.exit(0)
    is_handler_running_flag[0] = True
    print("You pressed Ctrl+C -- Running `signal_handler()` for graceful shutdown")
    # Save screen recording to disk
    screen_recorder.stop()
    screen_recorder.save(os.path.join(path_to_output_dir, f"{trace_name}.mp4"))
    sys.exit(0)


def setup_chrome_driver(is_headless: bool = False) -> webdriver.Chrome:
    """Attach Selenium driver to Chrome session running on port 9222"""
    print(f"Selenium is starting in {'headless' if is_headless else 'UI'} mode...")
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutoReload")
    options.debugger_address = "127.0.0.1:9222"
    options.headless = is_headless
    driver = webdriver.Chrome(options=options)
    print("Selenium is running...")
    return driver


def setup_playwright_driver(is_headless: bool = False) -> Browser:
    """Sping up Playwright instance"""
    print(f"Playwright is starting in {'headless' if is_headless else 'UI'} mode...")
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(channel="chrome", headless=is_headless)
    print("Playwright is running...")
    return playwright, browser


def parse_unknown_args_to_dict(unknown_args: List[str]) -> Dict[str, Any]:
    """Takes `unknown_args` output from parser.parse_known_args() and converts it to a dict"""
    iter_args = iter(unknown_args)  # Create an iterator
    unknown_args_dict = {}
    for arg in iter_args:
        if arg.startswith("-"):  # This is a flag
            next_arg = next(iter_args, None)
            if next_arg and not next_arg.startswith("-"):  # The next item is a value
                unknown_args_dict[arg.replace("-", "")] = next_arg
            else:  # The flag has no value
                unknown_args_dict[arg.replace("-", "")] = True
                if (
                    next_arg
                ):  # If there was a flag immediately following, process it in the next iteration
                    iter_args = iter([next_arg] + list(iter_args))

    return unknown_args_dict


def convert_dsl_to_actions(env, dsl: str) -> str:
    """Given an action in our DSL, convert it to the proper code for our environment `env`"""
    if env.is_headless:
        if env.env_type == "playwright":
            return convert_dsl_to_playwright_actions(dsl)
        elif env.env_type == "selenium":
            raise NotImplementedError("Selenium not supported yet")
        else:
            raise ValueError(f"Unknown env type: {env.env_type}")
    else:
        return convert_dsl_to_pyautogui_actions(dsl)


def extract_text_from_dsl_PRESS(dsl: str) -> str:
    start_removals = [
        "PRESS('",
        'PRESS("',
        "PRESS(",
    ]
    end_removals = [
        "')",
        '")',
        ")",
    ]
    for start in start_removals:
        if dsl.startswith(start):
            dsl = dsl[len(start) :]
    for end in end_removals:
        if dsl.endswith(end):
            dsl = dsl[: -len(end)]
    return dsl


def extract_text_from_dsl_TYPE(dsl: str) -> str:
    start_removals = [
        "TYPE('",
        'TYPE("',
        "TYPE(",
    ]
    end_removals = [
        "')",
        '")',
        ")",
    ]
    for start in start_removals:
        if dsl.startswith(start):
            dsl = dsl[len(start) :]
    for end in end_removals:
        if dsl.endswith(end):
            dsl = dsl[: -len(end)]
    return dsl


def extract_coords_from_dsl_CLICK(dsl: str) -> Tuple[float, float]:
    x, y = dsl.replace(" ", "").replace("CLICK(", "").replace(")", "").split(",")
    x = x.strip("'")
    y = y.strip("'")
    return float(x), float(y)


def extract_coords_from_dsl_SCROLL(dsl: str) -> Tuple[float, float]:
    dy = dsl.replace(" ", "").replace("SCROLL(", "").replace(")", "")
    return float(dy)


def convert_dsl_to_pyautogui_actions(dsl: str) -> str:
    """Given an action in our DSL, convert it to PyAutoGUI code"""
    dsl_steps = dsl.split("|")
    dsl_steps = [d.strip() for d in dsl_steps]
    script = f"""
import pyautogui
import time
time.sleep(1)
"""

    for dsl in dsl_steps:
        if dsl.startswith("CLICK"):
            x, y = extract_coords_from_dsl_CLICK(dsl)
            action: str = (
                f"pyautogui.moveTo(x={x}, y={y})\ntime.sleep(1)\npyautogui.click()"
            )
        elif dsl.startswith("TYPE"):
            text = extract_text_from_dsl_TYPE(dsl)
            action: str = f"pyautogui.typewrite('{text}')"
        elif dsl.startswith("PRESS"):
            text = extract_text_from_dsl_PRESS(dsl)
            action: str = f"pyautogui.press('{text}')"
        elif dsl.startswith("SCROLL"):
            dx = extract_coords_from_dsl_SCROLL(dsl)
            action: str = f"pyautogui.scroll({dx})"
        elif dsl.startswith("CLEAR"):
            action: str = f"pyautogui.hotkey('ctrl', 'a')\npyautogui.press('delete')"
        # elif dsl.startswith("NAVIGATE"):
        #     url = dsl.replace("NAVIGATE(", "").replace(")", "")
        #     action: str = f"env.get('{url}')"
        elif dsl.startswith("DELETE"):
            x, y = dsl.replace("DELETE(", "").replace(")", "").split(",")
            action: str = (
                f"pyautogui.moveTo(x={x}, y={y})\ntime.sleep(1)\npyautogui.click()\npyautogui.hotkey('command', 'a')\n"
            )

        script += "\n" + action

    return script


def convert_dsl_to_playwright_actions(dsl: str) -> str:
    """Given an action in our DSL, convert it to Playwright code"""
    if dsl.startswith("CLICK"):
        x, y = dsl.replace("CLICK(", "").replace(")", "").split(",")
        action: str = f"env.playwright_page.mouse.click({x}, {y})"
    elif dsl.startswith("TYPE"):
        text = (
            dsl.replace("TYPE(", "").replace(")", "").replace("'", "").replace('"', "")
        )
        action: str = f"env.playwright_page.keyboard.type('{text}')"
    elif dsl.startswith("PRESS"):
        text = (
            dsl.replace("PRESS(", "").replace(")", "").replace("'", "").replace('"', "")
        )
        if text.lower() in ["return", "enter"]:
            text = "Enter"
        if text.lower() in ["backspace", "delete"]:
            text = "Backspace"
        if text.lower() == "right":
            text = "ArrowRight"
        if text.lower() == "left":
            text = "ArrowLeft"
        if text.lower() == "up":
            text = "ArrowUp"
        if text.lower() == "down":
            text = "ArrowDown"
        action: str = f"env.playwright_page.keyboard.press('{text}')"
    elif dsl.startswith("SCROLL"):
        dx, dy = dsl.replace("SCROLL(", "").replace(")", "").split(",")
        action: str = f"env.playwright_page.mouse.wheel({dx}, {dy})"
    elif dsl.startswith("CLEAR"):
        action: str = f"env.playwright_page.keyboard.press('Meta+A')\nenv.playwright_page.keyboard.press('Delete')"
    # elif dsl.startswith("NAVIGATE"):
    #     url = dsl.replace("NAVIGATE(", "").replace(")", "")
    #     action: str = f"env.get('{url}')"
    else:
        print(f"ERROR - Unknown DSL: {dsl}")
        action: str = ""
    return f"""
import time
time.sleep(1)
{action}
time.sleep(1)
"""


def convert_pyautogui_to_playwright_actions(code: str) -> str:
    """Given a PyAutoGUI script (`code`), replace all `pyautogui` calls with Playwright calls"""
    playwright_code: List[str] = []
    for line in code.split("\n"):
        if not line.startswith("pyautogui"):
            # Leave all non-pyautogui code as-is
            playwright_code.append(line)
            continue
        line = line.replace("pyautogui.", "")  # remove prefix
        line = line.split("#")[0]  # remove comments
        if line.startswith("click"):
            # Click
            args: str = line.replace("click(", "").replace(")", "")
            args = [
                x.strip().replace("x=", "").replace("y=", "")
                for x in args.split(",")[:2]
                if x.strip() != ""
            ]  # ignore kwargs
            if len(args) == 0:
                # No (x,y)
                playwright_code.append(f"page.mouse.down()")
            else:
                playwright_code.append(f"page.mouse.click({','.join(args)})")
        elif line.startswith("moveTo"):
            # Move mouse
            args: str = line.replace("moveTo(", "").replace(")", "")
            args = [
                x.strip().replace("x=", "").replace("y=", "")
                for x in args.split(",")[:2]
            ]  # ignore kwargs
            playwright_code.append(f"page.mouse.move({','.join(args)})")
        elif line.startswith("scroll"):
            # Scroll
            args: str = line.replace("scroll(", "").replace(")", "")
            if int(args) < 0:
                # Scroll down
                args = f"0, {-int(args)}"
            else:
                # Scroll up
                args = f"{int(args)}, 0"
            playwright_code.append(f"page.mouse.wheel({args})")
        elif line.startswith("typewrite"):
            # Typing
            args: str = line.replace("typewrite(", "").replace(")", "")
            args = [x.strip() for x in args.split(",")[:1]]  # ignore kwargs
            playwright_code.append(f"page.keyboard.type({','.join(args)})")
        elif line.startswith("press"):
            # Keypress
            args: str = line.replace("press(", "").replace(")", "")
            args = args[1:-1]  # remove quotes around string
            args = args.capitalize()  # capitalize first letter for playwright
            if args == "Return":
                args = "Enter"
            playwright_code.append(f"page.keyboard.press('{args}')")
        elif line.startswith("hotkey"):
            # Hotkey (i.e. keypress)
            args: str = line.replace("hotkey(", "").replace(")", "")
            args = "+".join(
                [x.strip().replace("'", "").replace('"', "") for x in args.split(",")]
            )
            args = args.replace("ctrl", "Control")
            playwright_code.append(f"page.keyboard.press('{args}')")
    return "\n".join(playwright_code)


def get_rel_path(file: str, rel_path: str) -> str:
    """Transforms a relative path from a specific file in the package `eclair/src/eclair/ into an absolute path"""
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(file)), rel_path)
    )


def log(msg: str, path_to_log_file: str) -> None:
    print(msg)
    with open(path_to_log_file, "a") as fd:
        fd.write(f"{msg}\n")


def clean_gpt_code(code: str) -> str:
    """Does some basic preprocessing of GPT-4's output when asked to code"""
    code = code.replace("            ", "")
    try:
        code = code[code.index("import ") :]
        code = code[: code.index("```")]
    except:
        pass
    code = code.replace("```", "")
    return code


def run_code(env, code: str, logger: Callable[[str], None] = print) -> str:
    """Run `code` as Python"""
    try:
        if env.env_type == "playwright":
            # Needed for code
            page = env.playwright_page
        exec(code)
        return code
    except Exception as e:
        logger(f"Exception running code:\n```\n{code}\n```")
        logger(str(e))
        raise e


def replace_string_occurrences_in_dict(obj: Dict, target: str, replacement: str):
    """
    Recursively replaces occurrences of 'target' string with 'replacement' string in a dictionary.

    :param obj: The dictionary or list to be processed.
    :param target: The string to be replaced.
    :param replacement: The string to replace with.
    :return: The dictionary or list with replaced strings.
    """
    if isinstance(obj, dict):
        return {
            k: replace_string_occurrences_in_dict(v, target, replacement)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [
            replace_string_occurrences_in_dict(elem, target, replacement)
            for elem in obj
        ]
    elif isinstance(obj, str):
        return obj.replace(target, replacement)
    else:
        return obj


def find_highest_numbered_png_in_dir(path_to_dir: str) -> int:
    """
    Finds the highest numbered PNG file in a given folder.

    Args:
    path_to_dir (str): The path to the folder containing the PNG files.

    Returns:
    int: The highest number found among the PNG files. Returns -1 if no PNG files are found.
    """
    numbers = [
        int(f.split(".")[0])
        for f in os.listdir(path_to_dir)
        if f.endswith(".png") and f.split(".")[0].isdigit()
    ]
    return max(numbers) if len(numbers) > 0 else -1


def encode_image(path_to_img: str):
    """Base64 encode an image"""
    with open(path_to_img, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_screenshot_for_state(state: str, path_to_screenshots: str) -> Tuple[str, str]:
    path_to_screenshot: str = state["data"]["path_to_screenshot"]
    path_to_screenshot: str = path_to_screenshot.split("/")[-1]
    encoded_image: str = encode_image(
        os.path.join(path_to_screenshots, path_to_screenshot)
    )
    return path_to_screenshot, encoded_image


def add_grid_to_image(path_to_orig_img: str, path_to_new_img: str, grid_interval: int):
    """
    Adds a grid to an image for GPT-4 (x,y) coord locating. Draws lines every `grid_interval` pixels and adds labels to the lines.
    Source: https://github.com/OthersideAI/self-operating-computer/blob/main/operate/main.py
    """
    image = Image.open(path_to_orig_img)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Reduce the font size a bit
    font_size = int(grid_interval / 10)  # Reduced font size

    # Calculate the background size based on the font size
    bg_width = int(font_size * 4.2)  # Adjust as necessary
    bg_height = int(font_size * 1.2)  # Adjust as necessary

    # Function to draw text with a white rectangle background
    def draw_label_with_background(
        position, text, draw, font_size, bg_width, bg_height
    ):
        # Adjust the position based on the background size
        text_position = (position[0] + bg_width // 2, position[1] + bg_height // 2)
        # Draw the text background
        draw.rectangle(
            [position[0], position[1], position[0] + bg_width, position[1] + bg_height],
            fill="white",
        )
        # Draw the text
        draw.text(text_position, text, fill="black", font_size=font_size, anchor="mm")

    # Draw vertical lines and labels at every `grid_interval` pixels
    for x in range(grid_interval, width, grid_interval):
        line = ((x, 0), (x, height))
        draw.line(line, fill="blue")
        for y in range(grid_interval, height, grid_interval):
            # Calculate the percentage of the width and height
            x_percent = round((x / width) * 100)
            y_percent = round((y / height) * 100)
            draw_label_with_background(
                (x - bg_width // 2, y - bg_height // 2),
                f"{x_percent}%,{y_percent}%",
                draw,
                font_size,
                bg_width,
                bg_height,
            )

    # Draw horizontal lines - labels are already added with vertical lines
    for y in range(grid_interval, height, grid_interval):
        line = ((0, y), (width, y))
        draw.line(line, fill="blue")

    # Save the image with the grid
    image.save(path_to_new_img)


def collapse_key(data: List[Dict[str, Any]], key_to_collapse: str):
    """Given a list of dicts, convert the specified key's value to a compact JSON string
    Useful when pretty printing and want to compress `json_state`"""
    for item in data:
        if key_to_collapse in item:
            # Convert the specified key's value to a compact JSON string
            item[key_to_collapse] = json.dumps(item[key_to_collapse])
    return data


def get_last_element_attributes(env, key: Optional[str]) -> Optional[Dict[str, str]]:
    """Return the element attributes for the HTML element recorded by `event_listeners.js` given by `window.key` as parsed JSON dict."""
    if not key or key == "":
        return None
    elem: Optional[str] = env.execute_script(
        f"return window.{key} ? JSON.stringify(window.{key}) : localStorage.getItem('{key}');"
    )
    return json.loads(elem) if elem else None


def execute_js_scripts(env):
    """Helper function that re-executes a set of useful JavaScript scripts."""
    # CSS for proxy-select
    with open(
        get_rel_path(__file__, "./proxy-select/proxy-select.css"), "r"
    ) as fd:
        script: str = f"""
            var style = document.createElement('style');
            style.type = 'text/css';
            style.innerHTML = `{fd.read()}`;
            document.head.appendChild(style);
        """
        env.execute_script(script)
    # JS
    scripts: List[str] = [
        "./event_listeners.js",  # Map clicks/keystrokes to specific elements on the webpage
        "./proxy-select/proxy-select.js",  # Rerender dropdowns using proxy-select so that Playwright/Selenium can view them
    ]
    for path in scripts:
        with open(get_rel_path(__file__, path), "r") as fd:
            js_script: str = fd.read()
        env.execute_script(js_script)


def save_screenshot(path_to_screenshot: str, is_async: bool = False):
    """
    Takes a screenshot and saves it to `path_to_screenshot`
    Source: https://github.com/OthersideAI/self-operating-computer/blob/main/operate/main.py
    """
    user_platform: str = platform.system()
    if user_platform == "Windows":
        import pyautogui

        screenshot = pyautogui.screenshot()
        screenshot.save(path_to_screenshot)
    elif user_platform == "Linux":
        # Use xlib to prevent scrot dependency for Linux
        # screen = Xlib.display.Display().screen()
        # size = screen.width_in_pixels, screen.height_in_pixels
        # monitor_size["width"] = size[0]
        # monitor_size["height"] = size[1]
        # screenshot = ImageGrab.grab(bbox=(0, 0, size[0], size[1]))
        # screenshot.save(path_to_screenshot)
        raise ValueError("Linux not supported yet")
    elif user_platform == "Darwin":  # (Mac OS)
        # Use the screencapture utility to capture the screen with the cursor
        proc = subprocess.Popen(f'screencapture -C "{path_to_screenshot}"', shell=True)
        if not is_async:
            # Poll until screenshot is saved to file
            while proc.poll() is None:
                pass
    else:
        raise ValueError(
            f"The platform you're using ({user_platform}) is not currently supported"
        )


def fetch_openai_text_completion(
    prompt: str, system_prompt: Optional[str] = None, **kwargs
) -> str:
    """Helper function to call OpenAI's Text Completion API. Handles rate limit errors and other exceptions"""
    messages = [
        {
            "role": "system",
            "content": system_prompt if system_prompt is not None else SYSTEM_PROMPT,
        },
        {"role": "user", "content": prompt},
    ]
    return _fetch_openai_completion(messages, **kwargs)


def fetch_openai_json_completion(
    prompt: str, system_prompt: Optional[str] = None, **kwargs
) -> str:
    """Helper function to call OpenAI's JSON Completion API. Handles rate limit errors and other exceptions"""
    assert (
        "json" in prompt.lower()
    ), "Error - when using OpenAI JSON endpoint, your prompt must contain the word 'json' but currently doesn't"
    messages = [
        {
            "role": "system",
            "content": system_prompt if system_prompt is not None else SYSTEM_PROMPT,
        },
        {"role": "user", "content": prompt},
    ]
    return _fetch_openai_completion(
        messages,
        model=kwargs.get("model"),
        response_format={"type": "json_object"},
        **kwargs,
    )


def fetch_openai_vision_completion(
    prompt: str, base64_images: List[str], **kwargs
) -> str:
    """Helper function to call OpenAI's Vision API. Handles rate limit errors and other exceptions"""
    messages: List[Any] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                }
                for img in base64_images
            ]
            + [{"type": "text", "text": prompt}],
        },
    ]
    return _fetch_openai_completion(messages, model="gpt-4-vision-preview", **kwargs)


def is_coords_within_element(x: float, y: float, element: Dict[str, float]):
    """Return TRUE if (x,y) is in bbox of element
    Expects `element` to be dict with keys: x, y, height, width
    """
    box_x, box_y = element["x"], element["y"]
    box_width, box_height = element["width"], element["height"]
    return box_x <= x <= (box_x + box_width) and box_y <= y <= (box_y + box_height)


def find_sop_txt(path_to_task_dir: str) -> Optional[str]:
    """Find SOP.txt"""
    for file in os.listdir(path_to_task_dir):
        if file.startswith("SOP") and file.endswith(".txt"):
            return os.path.join(path_to_task_dir, file)
    return None


def find_gt_json(path_to_task_dir: str) -> Optional[str]:
    """Find [gt].json"""
    for file in os.listdir(path_to_task_dir):
        if file.startswith("[gt]") and file.endswith(".json"):
            return os.path.join(path_to_task_dir, file)
    return None


def adjust_json_state_xy_coords_to_center(json_state: Dict[str, str]) -> Dict[str, str]:
    """Given a `json_state`, loops through all elements and centers their (x,y) coords
    within the element, and removes `height` and `width` and old `x` and `y`"""
    for element in json_state:
        x, y = element["x"], element["y"]
        width, height = element["width"], element["height"]
        element["x"] = x + (width / 2)
        element["y"] = y + (height / 2)
        del element["width"]
        del element["height"]
    return json_state


def _fetch_openai_completion(messages: List[Any], model: str, **kwargs) -> str:
    """Helper function to call OpenAI's Vision API. Handles rate limit errors and other exceptions"""
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            model=model,
            max_tokens=4096,
            **kwargs,
        )
    except openai.RateLimitError:
        print("Rate limit exceeded -- waiting 1 min before retrying")
        time.sleep(60)
        return _fetch_openai_completion(messages, model, **kwargs)
    except openai.APIError as e:
        traceback.print_exc()
        print(f"OpenAI API error: {e}")
        sys.exit(1)
    except Exception as e:
        traceback.print_exc()
        print(f"Unknown error: {e}")
        sys.exit(1)
    return response.choices[0].message.content


def is_diff_base64_images(image_1: str, image_2: str) -> bool:
    """Return TRUE if images are NOT identical."""
    # screenshot_1 = Image.open(BytesIO(base64.b64decode(image_1)))
    # screenshot_2 = Image.open(BytesIO(base64.b64decode(image_2)))
    # diff = PIL.ImageChops.difference(screenshot_1, screenshot_2)
    # return diff.getbbox() is not None
    return image_1 != image_2


def run_validators(
    task_log: TaskLog, validators: List, is_keep_valid_feedback: bool = False
) -> Validation:
    """Given a list of Validators, run them in sequence and return results.
    If `is_keep_valid_feedback` is True, then the feedback from all Validators will be kept.
    Otherwise, just the invalid ones will be kept."""
    results: List[Validation] = []
    for validator in validators:
        validation: Validation = validator.run(task_log)
        results.append(validation)
    # Combine feedback
    return Validation(
        is_valid=all([result.is_valid for result in results]),
        feedback="\n".join(
            [
                f"- {result.feedback}"
                for result in results
                if is_keep_valid_feedback or not result.is_valid
            ]
        ),
        validations=results,
    )


def get_webarena_task_json(task_id: int) -> Optional[Dict[str, str]]:
    """Given the integer task ID, return the task JSON from the WebArena dataset"""
    path_to_webarena_tasks: str = get_rel_path(__file__, "../../data/vldb_experiments/webarena_tasks/")
    for filename in os.listdir(path_to_webarena_tasks):
        if not filename.endswith(".json") or filename.startswith("test"):
            # Skip non-JSON files and test files
            continue
        file_task_id: int = int(filename.split(".")[0])
        if int(task_id) == file_task_id:
            task = json.load(open(os.path.join(path_to_webarena_tasks, filename), "r"))
            return task
    return None


def convert_trace_action_to_dsl(event: Dict[str, str]) -> Dict[str, str]:
    """Converts a trace.json action into DSL format."""
    action_data: Dict[str, str] = {}
    tag_2_type = {
        "input": "text field",
        "button": "button",
        "a": "link",
        "select": "dropdown",
        None: "None",
    }
    if (
        "element_attributes" in event["data"]
        and event["data"]["element_attributes"] is not None
    ):
        if "element" in event["data"]["element_attributes"]:
            attrs: Dict[str, Any] = event["data"]["element_attributes"]["element"]
        else:
            attrs: Dict[str, Any] = event["data"]["element_attributes"]
        elem_x: int = int(attrs["x"]) if "x" in attrs else None
        elem_y: int = int(attrs["y"]) if "y" in attrs else None
        elem_height: int = int(attrs["height"]) if "height" in attrs else None
        elem_width: int = int(attrs["width"]) if "width" in attrs else None
        elem_text: str = attrs.get("text")
        elem_tag: str = attrs.get("tag")
        elem_label: str = attrs.get("label")
        elem_role: str = attrs.get("role")
        elem_placeholder: str = attrs.get("placeholder")
        elem_textified: str = (
            elem_label
            if elem_label
            else (
                elem_role
                if elem_role
                else (elem_text if elem_text else elem_placeholder)
            )
        )
        elem_type = tag_2_type[elem_tag] if elem_tag in tag_2_type else "text"
        elem_xpath: str = attrs.get("xpath")
    else:
        elem_x = None
        elem_y = None
        elem_height = None
        elem_width = None
        elem_text = None
        elem_tag = None
        elem_label = None
        elem_textified = None
        elem_type = None
        elem_xpath = None
    if event["data"]["type"] == "mouseup":
        action_data["action"] = f"Click on the {elem_type} labeled '{elem_textified}'"
        action_data["actuation_suggestion"] = {
            "action": f"CLICK({int(event['data']['x'])}, {int(event['data']['y'])})",
            "element": f"{{'x': {elem_x}, 'y': {elem_y}, 'height': {elem_height}, 'width': {elem_width}, 'text': '{elem_text}', 'tag': '{elem_tag}', 'xpath' : '{elem_xpath}' }}",
        }
    elif event["data"]["type"] in ["keystroke"]:
        keystroke: str = "".join(
            [x.replace("'", "") for x in event["data"]["key"].split(" ")]
        )
        keystroke = keystroke.replace("Key.space", " ")
        keystroke = keystroke.replace("Key.shift_r", "")
        keystroke = keystroke.replace("Key.shift", "")
        action_data["action"] = (
            f"Type the string '{keystroke}' in the {elem_type} labeled '{elem_textified}'"
        )
        action_data["actuation_suggestion"] = {
            "action": f'TYPE("{keystroke}")',
            "element": f"{{'x': {elem_x}, 'y': {elem_y}, 'height': {elem_height}, 'width': {elem_width}, 'text': '{elem_text}', 'tag': '{elem_tag}', 'xpath' : '{elem_xpath}'}}",
        }
    elif event["data"]["type"] in ["keypress"]:
        keystroke: str = "".join(
            [x.replace("'", "") for x in event["data"]["key"].split(" ")]
        )
        keystroke = keystroke.replace("Key.space", " ")
        keystroke = keystroke.replace("Key.", "")
        if keystroke.lower() in ["enter", "return"]:
            keystroke = "Enter"
        action_data["action"] = (
            f"Press the key '{keystroke}' in the {elem_type} labeled '{elem_textified}'"
        )
        action_data["actuation_suggestion"] = {
            "action": f'PRESS("{keystroke}")',
            "element": f"{{'x': {elem_x}, 'y': {elem_y}, 'height': {elem_height}, 'width': {elem_width}, 'text': '{elem_text}', 'tag': '{elem_tag}', 'xpath' : '{elem_xpath}'}}",
        }
    elif event["data"]["type"] == "scroll":
        action_data["action"] = ""
        if event["data"]["dy"] != 0:
            direction = "up" if event["data"]["dy"] < 0 else "down"
            pixels: int = abs(event["data"]["dy"])
            action_data["action"] += f"Scroll {direction} by {pixels} pixels."
        if event["data"]["dx"] != 0:
            direction = "left" if event["data"]["dx"] < 0 else "right"
            pixels: int = abs(event["data"]["dx"])
            action_data["action"] += (
                " " if event["data"]["dy"] != 0 else ""
            ) + f"Scroll {direction} by {pixels} pixels."
        action_data["actuation_suggestion"] = {
            "action": f'SCROLL({event["data"]["dx"]},{event["data"]["dy"]})',
            "element": "None",
        }
    else:
        raise ValueError(f"Unknown action type: {event['data']['type']}")
    return action_data


def is_coords_within_element(x: float, y: float, element: Dict[str, float]):
    """Return TRUE if (x,y) is in bbox of element
    Expects `element` to be dict with keys: x, y, height, width
    """
    box_x, box_y = element["x"], element["y"]
    box_width, box_height = element["width"], element["height"]
    return box_x <= x <= (box_x + box_width) and box_y <= y <= (box_y + box_height)


def find_sop_txt(path_to_task_dir: str) -> Optional[str]:
    """Find SOP.txt in a WebArena++ human demo folder"""
    for file in os.listdir(path_to_task_dir):
        if file.startswith("SOP") and file.endswith(".txt"):
            return os.path.join(path_to_task_dir, file)
    return None


def find_gt_json(path_to_task_dir: str) -> Optional[str]:
    """Find [gt].json in a WebArena++ human demo folder"""
    for file in os.listdir(path_to_task_dir):
        if file.startswith("[gt]") and file.endswith(".json"):
            return os.path.join(path_to_task_dir, file)
    return None


def adjust_json_state_xy_coords_to_center(json_state: Dict[str, str]) -> Dict[str, str]:
    """Given a `json_state`, loops through all elements and centers their (x,y) coords
    within the element, and removes `height` and `width` and old `x` and `y`"""
    for element in json_state:
        if "x" in element and "y" in element:
            x, y = element["x"], element["y"]
            width, height = element["width"], element["height"]
            element["x"] = x + (width / 2)
            element["y"] = y + (height / 2)
            del element["width"]
            del element["height"]
        else:
            print(element)
    return json_state

Inputs = namedtuple('Inputs', ['path_to_screenshots', 'path_to_gt_json', 'path_to_sop', 'gt_task_data', 'sop', 'model_kwargs'])
def load_files_for_task(path_to_task_dir: str) -> Inputs:
    path_to_screenshots: str = os.path.join(path_to_task_dir, "screenshots")
    path_to_gt_json: Optional[str] = find_gt_json(path_to_task_dir)
    path_to_sop: Optional[str] = find_sop_txt(path_to_task_dir)
    assert path_to_gt_json is not None, f"Could not find [gt].json file in {path_to_task_dir}"
    assert path_to_sop is not None, f"Could not find SOP.txt file in {path_to_task_dir}"
    
    # Read gt_trace.json
    gt_task_data: Dict[str, str] = json.load(open(path_to_gt_json, 'r'))

    # Read SOP.txt (if applicable)
    sop: Optional[str] = open(path_to_sop, 'r').read()

    # Load model
    model_kwargs = {
        "model": "gpt-4-vision-preview",
        "temperature": 0.0,
    }
    
    return Inputs(path_to_screenshots,
                    path_to_gt_json,
                    path_to_sop,
                    gt_task_data,
                    sop,
                    model_kwargs)

def find_files_by_prefix_suffix(
    directory, prefix="", suffix=".txt"
) -> List[str]:
    """Returns file name, not path"""
    matching_files = []
    for file in os.listdir(directory):
        if file.startswith(prefix) and file.endswith(suffix):
            matching_files.append(file)
    return matching_files

def get_path_to_screen_recording(path_to_task_folder: str, is_no_assert: bool = False) -> Optional[str]:
    files: List[str] = [
        x 
        for x in find_files_by_prefix_suffix(path_to_task_folder, "", ".mp4")
        if not (x.startswith("[gt]") or x.startswith("[raw]") or x.startswith("[clean]"))
    ]
    if not is_no_assert:
        assert len(files) > 0, f"Could not find any .mp4 files in {path_to_task_folder}"
    return os.path.join(path_to_task_folder, files[0]) if len(files) > 0 else None

def get_path_to_screenshots_dir(path_to_task_folder: str, is_no_assert: bool = False) -> str:
    file: str = os.path.join(path_to_task_folder, "screenshots/")
    if not is_no_assert:
        assert os.path.isdir(
            file
        ), f"Could not find screenshots directory in {path_to_task_folder}"
    return file

def get_path_to_trace_json(path_to_task_folder: str, is_no_assert: bool = False) -> Optional[str]:
    files: List[str] = [
        x
        for x in find_files_by_prefix_suffix(path_to_task_folder, "", ".json")
        if not (x.startswith("[gt]") or x.startswith("[raw]") or x.startswith("[clean]"))
    ]
    if not is_no_assert:
        assert (
            len(files) > 0
        ), f"Could not find any trace.json files in {path_to_task_folder}"
    return os.path.join(path_to_task_folder, files[0]) if len(files) > 0 else None

def extract_screenshots_for_demo(path_to_demo_folder: str, path_to_trace: Optional[str] = None, path_to_screen_recording: Optional[str] = None, is_verbose: bool = True):
    """Given a demo folder, extracts screenshots from the .mp4 screen recording and saves them to a `screenshots/` directory"""
    path_to_trace: str = get_path_to_trace_json(path_to_demo_folder) if not path_to_trace else path_to_trace
    path_to_screen_recording: str = get_path_to_screen_recording(path_to_demo_folder) if not path_to_screen_recording else path_to_screen_recording
    path_to_screenshots_dir: str = get_path_to_screenshots_dir(path_to_demo_folder)
    if os.path.exists(path_to_screenshots_dir):
        shutil.rmtree(path_to_screenshots_dir)
    os.makedirs(path_to_screenshots_dir)
    
    # Check that all necessary files/folders exist
    assert os.path.exists(path_to_demo_folder), f"Could not find {path_to_demo_folder}"
    assert os.path.exists(path_to_trace), f"Could not find trace.json in {path_to_demo_folder}"
    assert os.path.exists(path_to_screen_recording), f"Could not find screen recording in {path_to_demo_folder}"
    assert os.path.exists(path_to_screenshots_dir), f"Could not create screenshots directory in {path_to_demo_folder}"
  
    # Get .json trace
    full_trace: Dict[str, str] = json.loads(open(path_to_trace, "r").read())
    trace_json: List[Dict[str, str]] = full_trace['trace']

    with VideoFileClip(path_to_screen_recording) as video:
        img_idx: int = 0
        
        # Get start state's timestamps (for calculating secs_from_start later)
        start_state_timestamp: str = trace_json[0]['data']['timestamp']
        start_state_secs_from_start: str = trace_json[0]['data']['secs_from_start']
        # video_start_timestamp: datetime.datetime = datetime.datetime.fromtimestamp(
        #     os.stat(path_to_screen_recording).st_birthtime
        # )
        
        enumerator = tqdm(
            enumerate(trace_json),
            desc="Extracting Video => Screenshots",
            total=len([ x for x in trace_json if x['type'] == 'state' ]),
        ) if is_verbose else enumerate(trace_json)
        for event_idx, event in enumerator:
            if event["type"] == "state":
                # Our trace is: (S, A, S', A', ...)
                # For S, we want to take the screenshot immediately before A (for page loading / animations / etc.)
                # So we actually should ignore the time of S, and instead use slightly before the time of A for our extracted frame
                # (Except for the last S, which use its own time since there is no A after it)
                #
                # NOTE: We treat keystroke actions slightly differently than other actions given they last so long
                # We take the corresponding screenshot relative to `start_timestamp` and not `timestamp`
                is_next_action_keystroke: bool = (
                    event_idx + 1 < len(trace_json)
                    and trace_json[event_idx + 1]["type"] == "action"
                    and trace_json[event_idx + 1]["data"]["type"] == "keystroke"
                )
                timestamp: float = (
                    datetime.datetime.fromisoformat(
                        trace_json[
                            event_idx + 1
                            if len(trace_json) > event_idx + 1
                            else event_idx
                        ]["data"]["timestamp" if not is_next_action_keystroke else "start_timestamp"]
                    )
                    - datetime.datetime.fromisoformat(start_state_timestamp)
                ).total_seconds() + start_state_secs_from_start
                try:
                    if event_idx == len(trace_json) - 1:
                        # If final frame, leave 1s of buffer or use the last action's timestamp
                        frame = video.get_frame(min(timestamp, video.duration - 1))
                    else:
                        frame = video.get_frame(timestamp)
                    img: Image = Image.fromarray(frame)
                    img.save(os.path.join(path_to_screenshots_dir, f"{img_idx}.png"))
                    trace_json[event_idx]["data"][ "path_to_screenshot" ] = f"./screenshots/{img_idx}.png"
                    img_idx += 1
                except Exception as e:
                    print(
                        f"====> FAILED to extract screenshot: event_idx={event_idx} | timestamp={timestamp}. Exception: {e}"
                    )
            elif event["type"] == "action":
                # Make sure no screenshot associated with any actions
                if "path_to_screenshot" in event["data"]:
                    del trace_json[event_idx]["data"]["path_to_screenshot"]

    # Save updated trace with screenshot filenames
    n_screenshots: int = len(
        [x for x in os.listdir(path_to_screenshots_dir) if x.endswith(".png")]
    )
    assert n_screenshots == len(
        [x for x in trace_json if x["type"] == "state"]
    ), f"Number of screenshots ({n_screenshots}) does not match number of states ({len([ x for x in trace_json if x['type'] == 'state' ])})"
    full_trace['trace'] = trace_json
    json.dump(
        full_trace,
        open(path_to_trace, "w"),
        indent=2,
    )

def convert_mousedown_mouseup_to_click(trace_json: List[Dict[str, Any]], pixel_margin_of_error: float = 5.0, secs_margin_of_error: float = 2.0) -> List[Dict[str, Any]]:
    '''Merge consecutive mousedown/mouseup events at the same coords and within `secs_margin_of_error` secs of each other to a "CLICK" event
    This is lenient by +/- N pixels in x/y coords
    '''
    last_action: Dict[str, Any] = None
    idxs_to_remove: List[int] = []
    for idx, event in enumerate(trace_json):
        if event["type"] == "action":
            if (
                last_action is not None
                and last_action['data']['type'] == 'mousedown' 
                and event['data']['type'] == 'mouseup'
                and abs(last_action['data']['x'] - event['data']['x']) <= pixel_margin_of_error
                and abs(last_action['data']['y'] - event['data']['y']) <= pixel_margin_of_error
                and event['data']['secs_from_start'] - last_action['data']['secs_from_start'] <= secs_margin_of_error
            ):
                # We found two consecutive mousedown/mouseup events at the same(-ish) location
                last_action['data']['type'] = 'click'
                idxs_to_remove.append(idx)
            else:
                last_action = event
    return [event for idx, event in enumerate(trace_json) if idx not in idxs_to_remove]

def merge_consecutive_scrolls(trace_json: List[Dict[str, Any]], pixel_margin_of_error: float = 5.0) -> List[Dict[str, Any]]:
    '''Merge consecutive scroll events into a single scroll action.
    This is lenient by +/- N pixels in x/y coords
    '''
    last_action: Dict[str, Any] = None
    idxs_to_remove: List[int] = []
    for idx, event in enumerate(trace_json):
        if event["type"] == "action":
            if (
                last_action is not None
                and last_action['data']['type'] == 'scroll' 
                and event['data']['type'] == 'scroll'
                and abs(last_action['data']['x'] - event['data']['x']) <= pixel_margin_of_error
                and abs(last_action['data']['y'] - event['data']['y']) <= pixel_margin_of_error
            ):
                # We found two consecutive scroll events at the same(-ish) location
                last_action['data']['dx'] += event['data']['dx']
                last_action['data']['dy'] += event['data']['dy']
                idxs_to_remove.append(idx)
            else:
                last_action = event
    return [event for idx, event in enumerate(trace_json) if idx not in idxs_to_remove]

def merge_consecutive_states(trace_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Merge consecutive states into one state (the last one).'''
    idxs_to_remove: List[int] = []
    consecutive_event_idxs: List[int] = []
    for idx, event in enumerate(trace_json):
        if event["type"] == "state":
            consecutive_event_idxs.append(idx)
        if event['type'] != 'state' or idx == len(trace_json) - 1:
            # Found a non-state or at end of trace, so clear out our consecutive state tracker
            if len(consecutive_event_idxs) > 1:
                # keep the last state
                idxs_to_remove += consecutive_event_idxs[:-1]
            consecutive_event_idxs = []
    return [event for idx, event in enumerate(trace_json) if idx not in idxs_to_remove]

def remove_action_type(trace_json: List[Dict[str, Any]], action_type: str) -> List[Dict[str, Any]]:
    '''Remove all actions with type == `action_type`'''
    return [
        event 
        for event in trace_json 
        if (
            event['type'] != 'action' 
            or event['data']['type'] != action_type
        )
    ]

def remove_esc_key(trace_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Remove all keypresses with key == 'esc' '''
    return [
        event 
        for event in trace_json 
        if (
            event['type'] != 'action' 
            or event['data']['type'] not in ['keypress', 'keyrelease']
            or event['data']['key'] != 'Key.esc'
        )
    ]

def merge_consecutive_keystrokes(trace_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Merge consecutive keypresses/keyreleases into the same field into one atomic entry.'''
    last_action: Dict[str, Any] = None
    idxs_to_remove: List[int] = []
    prior_state: Dict[str, Any] = None # State immediately before current action
    prior_prior_state: Dict[str, Any] = None # State before last action (i.e. before the state immediately before to this action)
    for idx, event in enumerate(trace_json):
        if event["type"] == "state":
            prior_prior_state = prior_state
            prior_state = event['data']
        elif event["type"] == "action":
            if (
                last_action is not None # There is a previous action
                and last_action['data']['type'] in ['keypress', 'keyrelease', 'keystroke', ] # Previous action was key event
                and event['data']['type'] in ['keypress', 'keyrelease', 'keystroke',] # Current action is key event
                and prior_state['active_application_name'] == prior_prior_state['active_application_name'] # In same application
                and (
                    ( # If in web browser, then we need to be in the same HTML input field
                        prior_state['active_application_name'] in LIST_OF_BROWSER_APPLICATIONS # In web browser
                        and 'element_attributes' in last_action['data']
                        and 'element_attributes' in event['data']
                        and last_action['data']['element_attributes'] is not None
                        and event['data']['element_attributes'] is not None
                        and last_action['data']['element_attributes'].get('xpath', None) == event['data']['element_attributes'].get('xpath', None)
                    )
                    or ( # If not in web browser, then don't check HTML input field
                        prior_state['active_application_name'] not in LIST_OF_BROWSER_APPLICATIONS # Not in web browser
                    )
                )
                and (not event['data']['key'].startswith('Key.') or event['data']['key'] in ['Key.space', 'Key.shift', 'Key.shift_r', 'Key.caps_lock', 'Key.backspace']) # Ignore non-space/Shift special keys
            ):
                # We found two consecutive non-special-key keystroke events in the same HTML field (i.e. identical xpath)
                if event['data']['type'] == 'keypress':
                    # only record keypresses (i.e. ignore keyrelease events so that we don't double count keypresses)
                    last_action['data']['key'] += ' ' + event['data']['key']
                last_action['data']['type'] = 'keystroke' # merge into one atomic keystroke
                last_action['data']['end_timestamp'] = event['data']['timestamp']
                last_action['data']['timestamp'] = event['data']['timestamp'] # use end_timestamp as timestamp for this action, so that we know its finished by the time we record it as having "happened"; needed for long keystroke events
                last_action['data']['secs_from_start'] = event['data']['secs_from_start']
                idxs_to_remove.append(idx)
            else:
                last_action = event
                last_action['data']['start_timestamp'] = last_action['data']['timestamp']
    return [event for idx, event in enumerate(trace_json) if idx not in idxs_to_remove]


###############################################
###############################################
#
# Annotating images with bbox labels
#
###############################################
###############################################


@dataclass
class PredictedBbox:
    """Standard class for storing bboxes and their corresponding text labels for screenshots."""

    # Coordinates
    x: float # upper left
    y: float # upper left
    width: float
    height: float

    text: Optional[str] = None # e.g. "Medication Orders" or "Submit"
    tag: Optional[str] = None # e.g. button, input, etc.
    confidence: Optional[float] = None # e.g. 0.95
    
    def get_bbox_xyxy(self, is_flat: bool = False) -> Union[Tuple[int], List[Tuple[int, int]]]:
        """Return the bounding box coordinates in the format:
            - is_flat=True: [ x0, y0, x1, y1 ]
            - is_flat=False: [(x0, y0), (x1, y1)]
        """
        x0 = self.x
        y0 = self.y
        x1 = x0 + self.width
        y1 = y0 + self.height
        if is_flat:
            return [x0, y0, x1, y1]
        return [ (x0, y0), (x1, y1) ] 
    
    def get_bbox_xyxyxyxy(self, is_flat: bool = False) -> Union[Tuple[int], List[Tuple[int, int]]]:
        """Return the bounding box coordinates in the format:
            - is_flat=True: [x0, y0, x1, y1, x2, y2, x3, y3]
            - is_flat=False: [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        """
        x0 = self.x
        y0 = self.y
        x1 = x0 + self.width
        y1 = y0
        x2 = x1
        y2 = y1 + self.height
        x3 = x0
        y3 = y2
        if is_flat:
            return [x0, y0, x1, y1, x2, y2, x3, y3]
        return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]

def draw_boxes(image, preds: List[PredictedBbox], color='black', width=2):
    draw = ImageDraw.Draw(image)
    for pred in preds:
        p0, p1, p2, p3 = pred.get_bbox_xyxyxyxy(is_flat=False)
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

def get_text_size(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def draw_text(image, preds: List[PredictedBbox], color="black"):
    draw = ImageDraw.Draw(image)
    max_font_size: int = 60
    font_path: str = surya.postprocessing.text.get_font_path()
    for pred in preds:
        text: str = pred.text
        
        # Bbox
        bbox_width = pred.width
        bbox_height = pred.height
        box_font_size = max(6, min(int(.75 * bbox_height), max_font_size))
        
        # Font
        font = ImageFont.truetype(font_path, box_font_size)
        text_width, text_height = get_text_size(text, font)
        
        # Calculate font size
        while (text_width > bbox_width or text_height > bbox_height) and box_font_size > 6:
            box_font_size = box_font_size - 1
            font = ImageFont.truetype(font_path, box_font_size)
            text_width, text_height = get_text_size(text, font)

        # Calculate text position (centered in bbox)
        text_width, text_height = get_text_size(text, font)
        x = pred.x
        y = pred.y + (bbox_height - text_height) / 2

        draw.text((x, y), text, fill=color, font=font)

def save_image_annotated_with_bboxes(path_to_image: str, 
                                    path_to_output_dir: str, 
                                    preds: List[PredictedBbox], 
                                    is_bbox: bool = False,
                                    is_text: bool = False,
                                    is_bbox_text: bool = False,
                                    file_name_suffix: Optional[str] = None):
    """Given an image and a list of predicted bounding boxes, annotate the image with the bounding boxes and text labels."""
    name: str = os.path.basename(path_to_image).split(".")[0]
    name = f"{name}{'-' + str(file_name_suffix) if file_name_suffix else ''}"
    # Bboxes only
    if is_bbox:
        im = Image.open(path_to_image)
        draw_boxes(im, preds, color="black", width=4)
        im.save(os.path.join(path_to_output_dir, f"{name}-bbox.png"))
    # Text only
    if is_text:
        im = Image.new('RGB', im.size, color='white')
        draw_text(im, preds, color="red")
        im.save(os.path.join(path_to_output_dir, f"{name}-text.png"))
    # Bboxes and text
    if is_bbox_text:
        im = Image.open(path_to_image)
        draw_boxes(im, preds, color="black", width=4)
        draw_text(im, preds, color="red")
        im.save(os.path.join(path_to_output_dir, f"{name}-bbox-text.png"))

