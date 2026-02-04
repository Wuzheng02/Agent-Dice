import re
import json

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str=predicted_str.replace("[","").replace("]","")
    ground_truth_str=ground_truth_str.replace("[","").replace("]","")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
        predicted_token=list(predicted_tokens)[0]
        ground_truth_token=list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def parse_action_type(s):
    """Extract the action type (CLICK, TYPE, SCROLL, LONG_PRESS, OPEN_APP, PRESS_BACK, ...)"""
    s = s.strip()
    if s.startswith("CLICK"):
        return "CLICK"
    if s.startswith("TYPE"):
        return "TYPE"
    if s.startswith("SCROLL"):
        return "SCROLL"
    if s.startswith("LONG_PRESS"):
        return "LONG_PRESS"
    if s.startswith("OPEN_APP"):
        return "OPEN_APP"
    if s.startswith("PRESS_BACK"):
        return "PRESS_BACK"
    if s.startswith("PRESS_HOME"):
        return "PRESS_HOME"
    if s.startswith("PRESS_RECENT"):
        return "PRESS_RECENT"
    if s.startswith("ENTER"):
        return "ENTER"
    if s.startswith("WAIT"):
        return "WAIT"
    if s.startswith("COMPLETE"):
        return "COMPLETE"
    return None


def extract_point(s):
    """Extract [[x, y]] as tuple."""
    m = re.search(r"\[\s*\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]\s*\]", s)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def extract_bracket_content(s):
    """Extract content inside [...] for TYPE and OPEN_APP."""
    m = re.search(r"\[(.*)\]", s)
    if m:
        return m.group(1).strip()
    return ""


def relative_diff(p1, p2):
    """Compute relative diff = |p1−p2| / max(p2, 1e-6)"""
    return abs(p1 - p2) / max(abs(p2), 1e-6)


def single_eval(action, label):

    type_a = parse_action_type(action)
    type_l = parse_action_type(label)

    if type_a is None or type_l is None:
        return 0, 0


    
    group = {"PRESS_BACK", "PRESS_HOME", "PRESS_RECENT", "ENTER", "WAIT", "COMPLETE"}

    same_type = 0
    if type_a == type_l:
        same_type = 1

    if same_type == 0:
        return 0, 0

    if type_a in group:
        return 1, 1

    if type_a in {"CLICK", "LONG_PRESS"}:
        pa = extract_point(action)
        pl = extract_point(label)
        if pa is None or pl is None:
            return 1, 0 

        (xa, ya), (xl, yl) = pa, pl

        dx = abs(xa - xl) / 1000
        dy = abs(ya - yl) / 1000

        delta = (dx ** 2 + dy ** 2) ** 0.5
        if delta <= 0.14:
            return 1, 1
        else:
            return 1, 0

    if type_a == "SCROLL":
        return 1, int(action.strip() == label.strip())

    if type_a in {"TYPE", "OPEN_APP"}:
        ca = extract_bracket_content(action)
        cl = extract_bracket_content(label)

        f1 = calculate_f1_score(ca, cl)
        return 1, 1 if f1 >= 0.5 else 0

    return 1, 0
