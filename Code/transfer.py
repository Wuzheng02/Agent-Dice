import re
def transfer_qwen3vl2atlas(action):
    if action == "PRESS_HOME":
        return "PRESS_HOME"
    elif action == "PRESS_BACK":
        return "PRESS_BACK"
    elif action.startswith("TASK_COMPLETE"):
        return "COMPLETE"
    elif action.startswith("WAIT"):
        return "WAIT"
    elif action.startswith("SWIPE"):
        return action.replace("SWIPE", "SCROLL")
    elif action.startswith("TYPE"):
        return action.replace("TYPE", "TYPE ", 1)
    elif action.startswith("CLICK") or action.startswith("LONG_PRESS"):
        action_type = "CLICK" if action.startswith("CLICK") else "LONG_PRESS"
        coord_str = action[len(action_type):].strip("[]")
        x, y = map(int, coord_str.split(","))
        return f"{action_type} <point>[[{x},{y}]]</point>"
    return action
