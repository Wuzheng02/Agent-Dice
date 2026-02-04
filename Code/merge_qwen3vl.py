import torch
import os
from transformers import AutoModelForImageTextToText
import time
start_time = time.time()
# -------- config --------
PATH_A = "/data3/models_20251210/Qwen3-VL-8B-Instruct"
PATH_B1 = "/data3/wuzheng/OS-fusion-qwen3/AC"
PATH_B2 = "/data3/wuzheng/OS-fusion-qwen3/AITZ"
PATH_B3 = "/data3/wuzheng/OS-fusion-qwen3/Odyssey"
SAVE_PATH = "/data3/wuzheng/OS-fusion-qwen3/Agentrice-GUI-Qwen3VL"


os.makedirs(SAVE_PATH, exist_ok=True)

DTYPE = torch.bfloat16   

# -------- load --------
print("Loading models...")

model_A = AutoModelForImageTextToText.from_pretrained(
    PATH_A,
    torch_dtype=DTYPE,
    device_map="auto",
)

model_B1 = AutoModelForImageTextToText.from_pretrained(
    PATH_B1,
    torch_dtype=DTYPE,
    device_map="auto",
)

model_B2 = AutoModelForImageTextToText.from_pretrained(
    PATH_B2,
    torch_dtype=DTYPE,
    device_map="auto",
)

model_B3 = AutoModelForImageTextToText.from_pretrained(
    PATH_B3,
    torch_dtype=DTYPE,
    device_map="auto",
)

state_A = model_A.state_dict()
state_B1 = model_B1.state_dict()
state_B2 = model_B2.state_dict()
state_B3 = model_B3.state_dict()

merged_state = {}

# -------- utils --------
def softmax_three(a, b, c):
    """element-wise softmax for 3 tensors"""
    m = torch.max(torch.stack([a, b, c]), dim=0)[0]
    ea = torch.exp(a - m)
    eb = torch.exp(b - m)
    ec = torch.exp(c - m)
    s = ea + eb + ec
    return ea / s, eb / s, ec / s

def softmax_two(a, b):
    """element-wise softmax for 2 tensors"""
    m = torch.max(torch.stack([a, b]), dim=0)[0]
    ea = torch.exp(a - m)
    eb = torch.exp(b - m)
    s = ea + eb
    return ea / s, eb / s

# -------- fusion --------
print("Performing element-wise directional consensus fusion...")

for key in state_A:
    A = state_A[key]
    B1 = state_B1[key]
    B2 = state_B2[key]
    B3 = state_B3[key]

    # parameter deltas
    C1 = B1 - A
    C2 = B2 - A
    C3 = B3 - A

    # sign consistency
    s1 = C1 >= 0
    s2 = C2 >= 0
    s3 = C3 >= 0

    # init weights
    a = torch.zeros_like(A)
    b = torch.zeros_like(A)
    c = torch.zeros_like(A)

    # === case 1: all agree ===
    agree_all = (s1 == s2) & (s2 == s3)
    if agree_all.any():
        sa, sb, sc = softmax_three(C1.abs(), C2.abs(), C3.abs())
        a[agree_all] = sa[agree_all]
        b[agree_all] = sb[agree_all]
        c[agree_all] = sc[agree_all]

    # === case 2: pairwise agree ===
    agree_12 = (s1 == s2) & (s1 != s3)
    if agree_12.any():
        sa, sb = softmax_two(C1.abs(), C2.abs())
        a[agree_12] = sa[agree_12]
        b[agree_12] = sb[agree_12]

    agree_13 = (s1 == s3) & (s1 != s2)
    if agree_13.any():
        sa, sc = softmax_two(C1.abs(), C3.abs())
        a[agree_13] = sa[agree_13]
        c[agree_13] = sc[agree_13]

    agree_23 = (s2 == s3) & (s2 != s1)
    if agree_23.any():
        sb, sc = softmax_two(C2.abs(), C3.abs())
        b[agree_23] = sb[agree_23]
        c[agree_23] = sc[agree_23]

    # merge
    merged = A + a * C1 + b * C2 + c * C3
    merged_state[key] = merged.to(A.dtype)

    print(f"Merged: {key}")

# -------- save --------
print("Saving merged model...")
model_A.load_state_dict(merged_state)
model_A.save_pretrained(SAVE_PATH)

print(f"✅ Merged Qwen3-VL model saved to {SAVE_PATH}")
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
