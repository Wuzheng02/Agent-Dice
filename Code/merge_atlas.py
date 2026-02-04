import torch
from transformers import Qwen2VLForConditionalGeneration
import os
import time
start_time = time.time()
# -------- config --------
PATH_A = "/data3/models_20251210/OS-Atlas-Pro-7B"
PATH_B1 = "/data3/wuzheng/OS-fusion/AC"
PATH_B2 = "/data3/wuzheng/OS-fusion/AITZ"
PATH_B3 = "/data3/wuzheng/OS-fusion/Odyssey"
SAVE_PATH = "/data3/wuzheng/OS-fusion/Agentrice-GUI-atlas"
os.makedirs(SAVE_PATH, exist_ok=True)

# -------- load --------
model_A = Qwen2VLForConditionalGeneration.from_pretrained(PATH_A, torch_dtype="bfloat16", device_map="auto")
model_B1 = Qwen2VLForConditionalGeneration.from_pretrained(PATH_B1, torch_dtype="bfloat16", device_map="auto")
model_B2 = Qwen2VLForConditionalGeneration.from_pretrained(PATH_B2, torch_dtype="bfloat16", device_map="auto")
model_B3 = Qwen2VLForConditionalGeneration.from_pretrained(PATH_B3, torch_dtype="bfloat16", device_map="auto")

state_A = model_A.state_dict()
state_B1 = model_B1.state_dict()
state_B2 = model_B2.state_dict()
state_B3 = model_B3.state_dict()

merged_state = {}

def softmax_three(a, b, c):
    """ element-wise softmax for 3 tensors """
    m = torch.max(torch.stack([a, b, c]), dim=0)[0]
    ea = torch.exp(a - m)
    eb = torch.exp(b - m)
    ec = torch.exp(c - m)
    s = ea + eb + ec
    return ea / s, eb / s, ec / s

def softmax_two(a, b):
    """ element-wise softmax for 2 tensors """
    m = torch.max(torch.stack([a, b]), dim=0)[0]
    ea = torch.exp(a - m)
    eb = torch.exp(b - m)
    s = ea + eb
    return ea / s, eb / s

print("Performing element-wise parameter fusion...")

for key in state_A:
    A = state_A[key]
    B1 = state_B1[key]
    B2 = state_B2[key]
    B3 = state_B3[key]

    C1 = B1 - A
    C2 = B2 - A
    C3 = B3 - A

    # element-wise signs
    s1 = C1 >= 0
    s2 = C2 >= 0
    s3 = C3 >= 0

    # === case 1: all agree ===
    agree_all = (s1 == s2) & (s2 == s3)

    # prepare weights a,b,c
    a = torch.zeros_like(A)
    b = torch.zeros_like(A)
    c = torch.zeros_like(A)

    # --- softmax on all three ---
    if agree_all.any():
        sa, sb, sc = softmax_three(C1.abs(), C2.abs(), C3.abs())
        a[agree_all] = sa[agree_all]
        b[agree_all] = sb[agree_all]
        c[agree_all] = sc[agree_all]

    # === case 2: only some agree ===
    # mask for positions where C1,C2 agree but C3 disagree
    agree_12 = (s1 == s2) & (s1 != s3)
    if agree_12.any():
        sa, sb = softmax_two(C1.abs(), C2.abs())
        a[agree_12] = sa[agree_12]
        b[agree_12] = sb[agree_12]
        c[agree_12] = 0

    # mask for C1,C3 agree but C2 disagree
    agree_13 = (s1 == s3) & (s1 != s2)
    if agree_13.any():
        sa, sc = softmax_two(C1.abs(), C3.abs())
        a[agree_13] = sa[agree_13]
        b[agree_13] = 0
        c[agree_13] = sc[agree_13]

    # mask for C2,C3 agree but C1 disagree
    agree_23 = (s2 == s3) & (s2 != s1)
    if agree_23.any():
        sb, sc = softmax_two(C2.abs(), C3.abs())
        a[agree_23] = 0
        b[agree_23] = sb[agree_23]
        c[agree_23] = sc[agree_23]

    # compute final merged parameter
    merged = A + a * C1 + b * C2 + c * C3
    merged_state[key] = merged.to(A.dtype)

    print(f"Merged param: {key}")

print("Saving...")
model_A.load_state_dict(merged_state)
model_A.save_pretrained(SAVE_PATH)
print(f"Merged model saved to {SAVE_PATH}")
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
