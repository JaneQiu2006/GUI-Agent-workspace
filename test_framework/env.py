import os
from time import sleep
import random
from action_util import ActionType, ActionType, AndroidAction
import time
import base64
import traceback
from adb_bridge import AdbBridge, AdbError

try:
    from termcolor import colored, cprint
except ImportError:
    def colored(message, _color):
        return message

    def cprint(message):
        print(message)

_ADB_BRIDGE = None


def configure_adb(device_config=None):
    """Configure once; auto mode prefers a locally attached USB device."""
    global _ADB_BRIDGE
    device_config = device_config or {}
    _ADB_BRIDGE = AdbBridge(
        transport=device_config.get("transport", "auto"),
        serial=device_config.get("serial"),
        ssh_host=device_config.get("ssh_host"),
        ssh_port=device_config.get("ssh_port"),
        ssh_user=device_config.get("ssh_user"),
    )
    _ADB_BRIDGE.ensure_ready()
    return _ADB_BRIDGE


def get_adb_bridge():
    global _ADB_BRIDGE
    if _ADB_BRIDGE is None:
        _ADB_BRIDGE = configure_adb()
    return _ADB_BRIDGE


def adb_screenshot(temp_path, name):
    return get_adb_bridge().screenshot(os.path.join(temp_path, name))



def adb_input_text(text):
    base64_text = base64.b64encode(text.encode('utf-8')).decode('utf-8')
    get_adb_bridge().shell(
        "am", "broadcast", "-a", "ADB_INPUT_B64", "--es", "msg", base64_text
    )

def adb_back():
    get_adb_bridge().shell("input", "keyevent", "4")

def adb_go_home():
    get_adb_bridge().shell("input", "keyevent", "3")

def adb_enter():
    get_adb_bridge().shell("input", "keyevent", "66")

def adb_tap(x, y):
    get_adb_bridge().shell("input", "tap", int(x), int(y))

def adb_swipe(x1, y1, x2, y2, duration_ms=500):
    get_adb_bridge().swipe(x1, y1, x2, y2, duration_ms)

def escape_shell_text(text):
    chars_to_escape = ['\\','"', "'", '`', '$']
    for char in chars_to_escape:
        text = text.replace(char, '\\' + char)
    text = text.replace(" ", "%s")
    return text

class AndroidEmulator():
    def __init__(self, max_steps, temp_path, all_tasks = None, translate_action = None, save_images = False, task_id=0, sample_mode=None, device_config=None):
        """
        temp_path temporary path to store the images for evaluation
        """
        self.temp_path = temp_path
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.save_images = save_images
        self.image_id = str(time.time())
        self.terminated = False
        self.max_steps = max_steps
        self.steps = 0
        self.task_id = 0
        self.all_tasks = all_tasks
        if sample_mode == "random":
            # randomly sample a task from the task set
            self.current_task = random.choice(all_tasks)
        elif sample_mode == "sequential":
            self.current_task = all_tasks[self.task_id]
        else:
            print("Invalid sample mode")
        self.translate_action = translate_action
        self.history = []
        self.adb = configure_adb(device_config)
        self.screen_width, self.screen_height = self.adb.display_size()

    
    def terminate(self):
        # Returning home is enough for cleanup.  The old implementation force-stopped
        # every third-party app on the attached phone, which is unsafe for real devices.
        adb_go_home()

    def count_white_pixels(self, img):
        import numpy as np

        img = img.convert('RGB')
        data = np.array(img)
        white_count = np.sum(np.all(data > 240, axis=-1))
        return white_count > 2_300_000
    
    def get_obs(self):
        for _ in range(3):
            try:
                start_time = time.time()
                self.current_task = self.all_tasks[self.task_id]
                is_white = True
                imagepath = os.path.join(self.temp_path, f"{self.current_task}_{self.image_id}_{self.steps}.png")
                name=f"{self.current_task}_{self.image_id}_{self.steps}.png"
                adb_screenshot(self.temp_path,name)
                end_time = time.time()
                print("screenshot time:",end_time-start_time)
                return {
                        "task": self.current_task,
                        "image_path": imagepath,
                }          
            except Exception as e:
                print(f"Exception happened during screenshotting")
                print(e)
                print(traceback.format_exc())
                sleep(6)
                continue
        raise AdbError("连续三次设备截图失败")

    def step(self, raw_action: str):
        self.current_task = self.all_tasks[self.task_id]
        if self.terminated:
            return None
        try:
            action = self.translate_action(raw_action)
        except Exception as e:
            print(e)
            print(f"Failed to translate action: {raw_action}, terminating the environment")
            action = AndroidAction(action_type=ActionType.TaskImpossible)
        self.history.append(action)
        self.steps += 1
        if self.steps > self.max_steps:
            action = AndroidAction(action_type=ActionType.TaskImpossible)
            cprint(colored(f"Terminate the Emulator: Max Steps Exceeded {self.max_steps}.", "red"))
        screenshot = None
        info = {}

        start_time = time.time()
        for i in range(2):
            try:
                if action.action_type == ActionType.DualPoint:
                    assert len(action.touch_point) == 2
                    assert len(action.lift_point) == 2
                    if(action.touch_point!=action.lift_point):
                        adb_swipe(action.touch_point[0]*self.screen_width, action.touch_point[1]*self.screen_height, action.lift_point[0]*self.screen_width, action.lift_point[1]*self.screen_height)
                    else:
                        adb_tap(action.touch_point[0]*self.screen_width, action.touch_point[1]*self.screen_height)
                elif action.action_type == ActionType.Type:
                    adb_input_text(action.typed_text)
                elif action.action_type == ActionType.GoBack:
                    adb_back()
                elif action.action_type == ActionType.GoHome:
                    adb_go_home()
                elif action.action_type == ActionType.Enter:
                    adb_enter()
                elif action.action_type == ActionType.TaskComplete:
                    self.terminated = True
                elif action.action_type == ActionType.TaskImpossible:
                    self.terminated = True
                elif action.action_type == ActionType.WAIT:
                    pass
                elif action.action_type == ActionType.LONGPRESS:
                    adb_swipe(action.touch_point[0]*self.screen_width, action.touch_point[1]*self.screen_height, action.lift_point[0]*self.screen_width, action.lift_point[1]*self.screen_height)
                else:
                    raise Exception(f"Unknown action type: {action.action_type}")
                action_success = True
                screenshot = self.get_obs()
                break
            except Exception as e:
                cprint(colored("an Exception occurred during environment interaction", "red"))
                print(e)
                cprint(colored("Retrying", "red"))
                sleep(10)
                if i == 1:
                    action_success = False
                    info["error"] = str(e)
                    self.terminate()
                    return None
                continue
        end_time = time.time()
        print("step time:",end_time-start_time)
        if action.action_type == ActionType.TaskComplete:
            success = True
        else:
            success = False

        if success:
            self.terminated = True
        if self.terminated:
            self.terminate()
        return screenshot, self.terminated, success
