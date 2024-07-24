import datetime
import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple
import easyocr
import pyautogui
from eclair.utils.executors import BaseClass, Environment, get_active_application_state
from eclair.utils.helpers import (
    get_png_size,
    get_rel_path,
    PredictedBbox,
    save_image_annotated_with_bboxes,
)
from eclair.utils.logging import LIST_OF_BROWSER_APPLICATIONS, State


class Observer(BaseClass):
    """
    Purpose: Generate state representations (JSON, image, etc.) of the current screen
    """

    def __init__(
        self,
        env: Environment,
        path_to_screenshots_dir: str,
        is_take_screenshots: bool = True,
        is_save_intermediate_bbox_screenshots: bool = False,
        is_delete_xpath_from_json_state: bool = False,
    ):
        super().__init__(env=env)
        self.path_to_screenshots_dir: Optional[str] = path_to_screenshots_dir
        self.is_take_screenshots: bool = is_take_screenshots
        self.is_save_intermediate_bbox_screenshots: bool = (
            is_save_intermediate_bbox_screenshots
        )
        self.is_delete_xpath_from_json_state: bool = is_delete_xpath_from_json_state
        self.ocr_model = easyocr.Reader(["en"], gpu=True)  # OCR model for non-webpages
        # Sometimes, the width/height of the screenshots we take != the actual screen size
        # This is because of Mac Retina display, which doubles the resolution of the screen in screenshots
        # So, we need to rescale the coordinates to match the actual screen size
        # We calculate the rescale factors once, then save them for future use
        self.rescale_screenshot_width_factor: Optional[float] = None
        self.rescale_screenshot_height_factor: Optional[float] = None

    def run(self, is_take_screenshots: Optional[bool] = None) -> State:
        """Return state of application."""
        # Get screenshot
        path_to_screenshot: Optional[str] = None
        if self.is_take_screenshots and (
            is_take_screenshots is None or is_take_screenshots
        ):
            path_to_screenshot = os.path.join(
                self.path_to_screenshots_dir,
                f"{int(datetime.datetime.now().timestamp() * 1000)}.png",
            )
            try:
                self.env.save_screenshot(path_to_screenshot, is_async=False)
            except Exception as e:
                print(str(e))
                print(f"Error taking screenshot for path: `{path_to_screenshot}`")

        # Get active application window specs
        active_application_state: Dict[str, Any] = get_active_application_state(
            self.env
        )
        is_application_browser: bool = (
            active_application_state["name"] in LIST_OF_BROWSER_APPLICATIONS
        )

        # Get state as either (a) JSON list of elements (if webpage) or (b) OCR'd text (if not webpage)
        url: Optional[str] = None
        tab: Optional[str] = None
        json_state: Optional[List[Dict[str, str]]] = None
        pred_bboxes: List[PredictedBbox] = []
        is_application_browser = True
        if is_application_browser:
            # Webpage, so get JSON state from JS
            json_state = self.convert_webpage_to_json_elements(self.env)
            url = self.env.current_url
            tab = self.env.title
            # No rescaling needed for HTML-derived JSON state b/c we're not pulling coords from a screenshot
            self.rescale_screenshot_width_factor = 1.0
            self.rescale_screenshot_height_factor = 1.0
        else:
            # Non-webpage, so get OCR'd text
            ocr_preds: List[Tuple[List[int], str, float]] = self.ocr_model.readtext(
                path_to_screenshot
            )
            pred_bboxes: List[PredictedBbox] = [
                PredictedBbox(
                    # NOTE: Casting from np.int64 -> int makes life easier for downstream JSON serialization code
                    x=int(bbox[0][0]),
                    y=int(bbox[0][1]),
                    width=int(bbox[2][0] - bbox[0][0]),
                    height=int(bbox[2][1] - bbox[0][1]),
                    text=text,
                    confidence=confidence,
                    tag="clickable",
                )
                for (bbox, text, confidence) in ocr_preds
            ]
            json_state = [
                {
                    # Center (x,y) within element
                    "x": pred.x + pred.width / 2,
                    "y": pred.y + pred.height / 2,
                    "height": pred.height,
                    "width": pred.width,
                    "tag": "button",
                    "text": "",
                    "type": "button",
                    "label": pred.text,
                    "role": "clickable",
                    "xpath": None,
                }
                for pred in pred_bboxes
            ]

        if self.is_save_intermediate_bbox_screenshots:
            # Save screenshot with bounding boxes + OCR'd text
            save_image_annotated_with_bboxes(
                path_to_screenshot,
                os.path.dirname(path_to_screenshot),
                pred_bboxes,
                is_bbox_text=True,
            )

        # NOTE: Due to Mac Retina display, the screenshot is 2x the size of the actual screen
        # So, we need to rescale the coordinates to match the actual screen size
        if (
            self.rescale_screenshot_width_factor is None
            or self.rescale_screenshot_height_factor is None
        ):
            # Only need to calculate this once, then save it for future use
            screen_display_size = pyautogui.size()
            screenshot_w, screenshot_h = get_png_size(path_to_screenshot)
            self.rescale_screenshot_width_factor = (
                screen_display_size.width / screenshot_w
            )
            self.rescale_screenshot_height_factor = (
                screen_display_size.height / screenshot_h
            )
        for elem in json_state:
            elem["x"] = int(elem["x"] * self.rescale_screenshot_width_factor)
            elem["y"] = int(elem["y"] * self.rescale_screenshot_height_factor)
            elem["width"] = int(elem["width"] * self.rescale_screenshot_width_factor)
            elem["height"] = int(elem["height"] * self.rescale_screenshot_height_factor)

        return State(
            url=url,
            tab=tab,
            json_state=json_state,
            html=self.env.content() if is_application_browser else None,
            screenshot_base64=None,  # must be set later
            path_to_screenshot=path_to_screenshot,
            active_application_name=active_application_state["name"],
            window_size={
                k: v
                for k, v in active_application_state.items()
                if k in ["width", "height"]
            },
            window_position={
                k: v for k, v in active_application_state.items() if k in ["x", "y"]
            },
            screen_size={
                "width": pyautogui.size().width,
                "height": pyautogui.size().height,
            },
            is_headless=self.env.is_headless,
        )

    def convert_webpage_to_json_elements(
        self,
        env: Environment,
    ) -> List[Dict[str, str]]:
        """Converts the current webpage into a JSON list of dicts, where each dict is a visible/relevant HTML element and their attributes."""
        if env.env_type == "desktop":
            return {}
        # Get current state as JSON blob
        with open(
            "/Users/avanikanarayan/Developer/Research/big-brother/eclair-agents/eclair/utils/get_webpage_state.js",
            "r",  ##get_rel_path(__file__, "../../../utils/get_webpage_state.js"), "r"
        ) as fd:
            js_script: str = fd.read()
        json_state: Dict[str, str] = json.loads(env.execute_script(js_script))

        # Adjust (x,y) coordinates to account for browser window position
        browser_width: int = env.execute_script(
            "return window.outerWidth;"
        )  # width of browser window
        browser_viewport_width: int = env.execute_script(
            "return window.innerWidth;"
        )  # width of webpage itself
        browser_height: int = env.execute_script(
            "return window.outerHeight;"
        )  # height of browser window
        browser_viewport_height: int = env.execute_script(
            "return window.innerHeight;"
        )  # height of webpage itself
        browser_chrome_width: int = browser_width - browser_viewport_width
        browser_chrome_height: int = browser_height - browser_viewport_height
        browser_coords: Dict[str, int] = (
            env.get_window_rect()
        )  # coords of browser on screen
        browser_x, browser_y = browser_coords["x"], browser_coords["y"]
        for element in json_state:
            # Account for positioning of browser window on screen
            if env.is_headless:
                # Ignore chrome since is_headless ignores chrome (i.e. positions (0,0) as top-left of viewport)
                element["x"] += browser_x
                element["y"] += browser_y
            else:
                element["x"] += browser_x + browser_chrome_width
                element["y"] += browser_y + browser_chrome_height
            # Adjust (x,y) to be slightly more within element (to be on the safer side when actuating)
            # Center (x,y) within element
            element["x"] += element["width"] / 2
            element["y"] += element["height"] / 2

        # Drop xpath
        for element in json_state:
            if self.is_delete_xpath_from_json_state:
                del element["xpath"]
            for key in [
                "role",
                "text",
                "type",
                "label",
                "is_focused",
                "is_disabled",
                "is_checked",
            ]:
                if key in element and element[key] is None or element[key] == False:
                    del element[key]

        # Add chrome omnibox to state (for navigation)
        if not self.env.is_headless:
            chrome_omnibox: Dict[str, Any] = {
                "x": browser_coords["x"] + 250,
                "y": browser_coords["y"] + browser_chrome_height / 2 + 20,
                "height": browser_chrome_height,
                "width": browser_width,
                "tag": "chrome_omnibox",
                "role": "Address bar / Search box",
                "text": "",
                "type": "input",
                "label": "Chrome Omnibox - This is an address bar that can be used to search a query on Google or navigate to a URL",
            }
            json_state.append(chrome_omnibox)

        return json_state
