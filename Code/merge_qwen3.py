import torch
import os
from transformers import AutoModelForCausalLM
import time


start_time = time.time()
# -------- config --------
PATH_A  = "/data1/model_20251218/Qwen3-8B-model"
PATH_B1 = "/data1/home/wuzheng/Agent_DiCE_API/models/subset0"
PATH_B2 = "/data1/home/wuzheng/Agent_DiCE_API/models/subset1"
PATH_B3 = "/data1/home/wuzheng/Agent_DiCE_API/models/subset2"
PATH_B4 = "/data1/home/wuzheng/Agent_DiCE_API/models/subset3"

SAVE_PATH = "/data1/home/wuzheng/Agent_DiCE_API/models/Qwen3-8B-fused"

os.makedirs(SAVE_PATH, exist_ok=True)

DTYPE = torch.bfloat16 

# -------- load --------
print("Loading models...")

def load(path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=DTYPE,
        device_map="balanced",
    )

model_A  = load(PATH_A)
model_B1 = load(PATH_B1)
model_B2 = load(PATH_B2)
model_B3 = load(PATH_B3)
model_B4 = load(PATH_B4)

state_A  = model_A.state_dict()
state_B1 = model_B1.state_dict()
state_B2 = model_B2.state_dict()
state_B3 = model_B3.state_dict()
state_B4 = model_B4.state_dict()

merged_state = {}

# -------- utils --------
def softmax_n(tensors):
    """
    element-wise softmax over N tensors
    """
    stack = torch.stack(tensors, dim=0)
    m = stack.max(dim=0)[0]
    exp = torch.exp(stack - m)
    s = exp.sum(dim=0)
    return [exp[i] / s for i in range(len(tensors))]

# -------- fusion --------
print("Performing 4-model directional consensus fusion (LM)...")

for key in state_A:
    A = state_A[key]

    C = [
        state_B1[key] - A,
        state_B2[key] - A,
        state_B3[key] - A,
        state_B4[key] - A,
    ]

    signs = [c >= 0 for c in C]
    sign_sum = sum(s.int() for s in signs)  # ∈ {0..4}

    weights = [torch.zeros_like(A) for _ in range(4)]

    # ===== case 1: ≥3 agree =====
    for target_sum in (3, 4, 1, 0):
        mask = sign_sum == target_sum
        if not mask.any():
            continue

        if target_sum >= 3:
            idx = [i for i in range(4) if signs[i].any()]
        else:
            idx = [i for i in range(4) if (~signs[i]).any()]

        mags = [C[i].abs() for i in idx]
        ws = softmax_n(mags)

        for i, w in zip(idx, ws):
            weights[i][mask] = w[mask]

    # ===== case 2: exactly 2 vs 2 =====
    tie_2v2 = sign_sum == 2
    if tie_2v2.any():
        mags = [c.abs() for c in C]
        ws = softmax_n(mags)
        for i in range(4):
            weights[i][tie_2v2] = ws[i][tie_2v2]

    # merge
    merged = A
    for w, c in zip(weights, C):
        merged = merged + w * c

    merged_state[key] = merged.to(A.dtype)

    print(f"Merged: {key}")

# -------- save --------
print("Saving fused model...")
model_A.load_state_dict(merged_state)
model_A.save_pretrained(SAVE_PATH)

print(f"✅ Fused Qwen3-8B model saved to {SAVE_PATH}")
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
