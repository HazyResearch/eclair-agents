import time
from typing import Callable, Dict, Optional, Any, List
from selenium import webdriver
from selenium.webdriver.common.by import By
from eclair.utils.helpers import save_screenshot, setup_chrome_driver, setup_playwright_driver
from AppKit import NSWorkspace
import Quartz

class Environment:
    """
        Wrapper around selenium + playwright + desktop
        Tries to conform to Selenium API as closely as possible.
        
        NOTE: `desktop` is useful when we know we're only using a desktop application (e.g. Epic)
                and don't need to interact with a browser. Thus, every function becomes a no-op.
    """

    ALLOWED_ENVS: List[str] = ["selenium", "playwright", "desktop"]

    def __init__(self, 
                 env_type: str = "selenium"):
        self.env_type: str = env_type
        
        assert env_type in self.ALLOWED_ENVS, f"Invalid env_type: {env_type}"
    
    def start(self, *args, is_headless: bool = False, record_video_dir: Optional[str] = None, **kwargs):
        """Creates a new browser instance (if applicable)."""
        self.is_headless: bool = is_headless if self.env_type != "desktop" else False # NOTE: `desktop` is always non-headless
        if self.env_type == "selenium":
            self.selenium_driver: webdriver.Chrome = setup_chrome_driver(*args, is_headless=is_headless, **kwargs)
        elif self.env_type == "playwright":
            self.playwright, self.playwright_browser = setup_playwright_driver(*args, is_headless=is_headless, **kwargs)
            self.playwright_context = self.playwright_browser.new_context(record_video_dir=record_video_dir)
            self.playwright_page = self.playwright_context.new_page()
        elif self.env_type == "desktop":
            pass
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")
            
    def stop(self):
        if self.env_type == "selenium":
            self.selenium_driver.quit()
        elif self.env_type == "playwright":
            self.playwright_context.close()
            self.playwright_browser.close()
            self.playwright.stop()
        elif self.env_type == "desktop":
            pass
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")

    def reset(self):
        if self.env_type == "selenium":
            self.selenium_driver.delete_all_cookies()
        elif self.env_type == "playwright":
            self.playwright_context.close()
            self.playwright_context = self.playwright_browser.new_context()
            self.playwright_page = self.playwright_context.new_page()
        elif self.env_type == "desktop":
            pass
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")

    def get(self, url: str):
        """Navigates to the given URL."""
        if self.env_type == "selenium":
            self.selenium_driver.get(url)
        elif self.env_type == "playwright":
            self.playwright_page.goto(url)
        elif self.env_type == "desktop":
            pass
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")
    
    def find_elements(self, css_selector: str) -> List[Any]:
        if self.env_type == "selenium":
            return self.selenium_driver.find_elements(By.CSS_SELECTOR, css_selector)
        elif self.env_type == "playwright":
            return self.playwright_page.query_selector_all(css_selector)
        elif self.env_type == "desktop":
            return []
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")
    
    def find_element(self, css_selector: str) -> Optional[Any]:
        if self.env_type == "selenium":
            return self.selenium_driver.find_element(By.CSS_SELECTOR, css_selector)
        elif self.env_type == "playwright":
            return self.playwright_page.query_selector(css_selector)
        elif self.env_type == "desktop":
            return None
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")
    
    def type_in_element(self, css_selector: str, text: str):
        """Types `text` in the element specified by `css_selector`."""
        if self.env_type == "selenium":
            self.find_element(css_selector).send_keys(text)
        elif self.env_type == "playwright":
            self.find_element(css_selector).fill(text)
        elif self.env_type == "desktop":
            pass
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")

    def click_element(self, css_selector: str):
        """Clicks the element specified by `css_selector`."""
        if self.env_type == "selenium":
            self.find_element(css_selector).click()
        elif self.env_type == "playwright":
            self.find_element(css_selector).click()
        elif self.env_type == "desktop":
            pass
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")

    @property
    def current_url(self) -> str:
        """Get current URL."""
        if self.env_type == "selenium":
            return self.selenium_driver.current_url
        elif self.env_type == "playwright":
            return self.playwright_page.url
        elif self.env_type == "desktop":
            return ""
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")
    
    @property
    def title(self) -> str:
        """Get current tab name."""
        if self.env_type == "selenium":
            return self.selenium_driver.title
        elif self.env_type == "playwright":
            return self.playwright_page.title()
        elif self.env_type == "desktop":
            return ""
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")
    
    def get_window_rect(self) -> Dict[str, int]:
        """Get coordinates of browser on screen."""
        if self.env_type == "selenium":
            return self.selenium_driver.get_window_rect()
        elif self.env_type == "playwright":
            return get_active_application_state(self)
        elif self.env_type == "desktop":
            return {}
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")
    
    def content(self) -> str:
        """Gets the full HTML contents of the page, including the doctype"""
        if self.env_type == "selenium":
            return self.selenium_driver.page_source
        elif self.env_type == "playwright":
            return self.playwright_page.content()
        elif self.env_type == "desktop":
            return ""
        else:
            raise Exception(f"Invalid env_type: {self.env_type}")

    def execute_script(self, script: str, is_playwright_use_wrapper: bool = True, is_retry: bool = True) -> str:
        """Executes JS script on the current webpage"""
        try:
            if self.env_type == "selenium":
                return self.selenium_driver.execute_script(script)
            elif self.env_type == "playwright":
                # Note: For playwright, we need to inject () => {}
                return self.playwright_page.evaluate(f"() => {{ {script} }}" if is_playwright_use_wrapper else script)
            elif self.env_type == "desktop":
                pass
            else:
                raise Exception(f"Invalid env_type: {self.env_type}")
        except Exception as e:
            print("Error in execute_script()")
            print(f"Tried to execute: `{script}`")
            print(f"Exception: {e}")
            if is_retry:
                print("Retrying in 5 seconds...")
                time.sleep(5)
                return self.execute_script(script, is_playwright_use_wrapper=is_playwright_use_wrapper, is_retry=False)
            return None
    
    def save_screenshot(self, path_to_output: str, is_async: bool = False):
        """Saves screenshot to `path_to_output`"""
        if self.is_headless:
            if self.env_type == "selenium":
                self.selenium_driver.save_screenshot(path_to_output)
            elif self.env_type == "playwright":
                # NOTE: PlayWright doesn't capture <SELECT> menus in screenshots (b/c handled by OS)
                # Thus, we need to inject our own version of <SELECT> menus; this must have been previously
                # done by calling `execute_js_scripts(env)`
                self.playwright_page.screenshot(path=path_to_output)
            else:
                raise Exception(f"Invalid env_type: {self.env_type}")
        else:
            save_screenshot(path_to_output, is_async=is_async)

class BaseClass:
    def __init__(self, 
                 model_kwargs: Optional[Dict[str, str ]] = None, 
                 env: Optional[Environment] = None):
        self.logger: Callable = lambda x : x
        self.model_kwargs: Dict[str, str] = model_kwargs or {}
        self.env: Optional[Environment] = env

    def set_logger(self, logger: Callable):
        """Define the function that we'll call to log things."""
        self.logger = logger

TaskLog = None
Validation = None
class BaseValidator(BaseClass):
    """
    Base class for all validators
    """
    def run(self, task_log: TaskLog) -> Validation:
        raise NotImplementedError


def get_active_application_state(env: Environment) -> Dict[str, Any]:
    """Get the name of the currently active desktop application."""
    if env.is_headless:
        if env.env_type == "playwright":
            return {
                "name": "Google Chrome",
                "x": 0,
                "y": 0,
                "width": env.playwright_page.viewport_size["width"],
                "height": env.playwright_page.viewport_size["height"],
            }
        elif env.env_type == "selenium":
            return {
                "name": "Google Chrome",
                "x": 0,
                "y": 0,
                "width": env.driver.get_window_size()["width"],
                "height": env.driver.get_window_size()["height"],
            }
        else:
            raise Exception(f"Error - Unsupported env_type: `{env.env_type}`")
    active_app_name: str = NSWorkspace.sharedWorkspace().activeApplication()[
        "NSApplicationName"
    ]
    window_info_list: List = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    x, y, width, height = None, None, None, None
    for window_info in window_info_list:
        if window_info["kCGWindowOwnerPID"] == 0:
            x: int = int(window_info["kCGWindowBounds"]["X"])
            y: int = int(window_info["kCGWindowBounds"]["Y"])
            width = int(window_info["kCGWindowBounds"]["Width"])
            height = int(window_info["kCGWindowBounds"]["Height"])
            break

    return {
        "name": active_app_name,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }
