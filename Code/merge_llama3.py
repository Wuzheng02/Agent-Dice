import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
start_time = time.time()
# -------- config --------
PATH_BASE = "/data3/models_20251210/Llama-3.1-8B" 
PATH_S0 = "/data3/wuzheng/Agent-DiCE-tool-llama3/subset0"
PATH_S1 = "/data3/wuzheng/Agent-DiCE-tool-llama3/subset1"
PATH_S2 = "/data3/wuzheng/Agent-DiCE-tool-llama3/subset2"
PATH_S3 = "/data3/wuzheng/Agent-DiCE-tool-llama3/subset3"

SAVE_PATH = "/data3/wuzheng/Agent-DiCE-tool-llama3/Agent-DiCE-llama3"

os.makedirs(SAVE_PATH, exist_ok=True)

DTYPE = torch.bfloat16 

# -------- utils --------
def softmax_four(a, b, c, d):
    """element-wise softmax for 4 tensors"""
    stack = torch.stack([a, b, c, d])
    m = torch.max(stack, dim=0)[0]
    
    ea = torch.exp(a - m)
    eb = torch.exp(b - m)
    ec = torch.exp(c - m)
    ed = torch.exp(d - m)
    
    s = ea + eb + ec + ed
    s = s + 1e-6
    return ea / s, eb / s, ec / s, ed / s

def softmax_three(a, b, c):
    """element-wise softmax for 3 tensors"""
    stack = torch.stack([a, b, c])
    m = torch.max(stack, dim=0)[0]
    
    ea = torch.exp(a - m)
    eb = torch.exp(b - m)
    ec = torch.exp(c - m)
    
    s = ea + eb + ec + 1e-6
    return ea / s, eb / s, ec / s

# -------- load --------
print("Loading models (This may take a while due to RAM usage)...")

print(f"Loading Base: {PATH_BASE}")
model_base = AutoModelForCausalLM.from_pretrained(PATH_BASE, torch_dtype=DTYPE, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(PATH_BASE) 

print(f"Loading Subset 0: {PATH_S0}")
model_s0 = AutoModelForCausalLM.from_pretrained(PATH_S0, torch_dtype=DTYPE, device_map="auto")
print(f"Loading Subset 1: {PATH_S1}")
model_s1 = AutoModelForCausalLM.from_pretrained(PATH_S1, torch_dtype=DTYPE, device_map="auto")

print(f"Loading Subset 2: {PATH_S2}")
model_s2 = AutoModelForCausalLM.from_pretrained(PATH_S2, torch_dtype=DTYPE, device_map="auto")

print(f"Loading Subset 3: {PATH_S3}")
model_s3 = AutoModelForCausalLM.from_pretrained(PATH_S3, torch_dtype=DTYPE, device_map="auto")

state_base = model_base.state_dict()
state_s0 = model_s0.state_dict()
state_s1 = model_s1.state_dict()
state_s2 = model_s2.state_dict()
state_s3 = model_s3.state_dict()

merged_state = {}

# -------- fusion --------
print("\nPerforming 4-way Directional Consensus Fusion...")
print("Strategy: 4:0 (Keep All), 2:2 (Keep All), 3:1 (Keep Majority)")

with torch.no_grad():
    for key in state_base:
        if key not in state_s0: 
            merged_state[key] = state_base[key]
            continue

        W_base = state_base[key]
        W_0 = state_s0[key]
        W_1 = state_s1[key]
        W_2 = state_s2[key]
        W_3 = state_s3[key]

        C0 = W_0 - W_base
        C1 = W_1 - W_base
        C2 = W_2 - W_base
        C3 = W_3 - W_base

        s0 = C0 >= 0
        s1 = C1 >= 0
        s2 = C2 >= 0
        s3 = C3 >= 0

        sum_signs = s0.long() + s1.long() + s2.long() + s3.long()

        w0 = torch.zeros_like(W_base)
        w1 = torch.zeros_like(W_base)
        w2 = torch.zeros_like(W_base)
        w3 = torch.zeros_like(W_base)

        mask_use_all = (sum_signs == 4) | (sum_signs == 0) | (sum_signs == 2)
        
        if mask_use_all.any():
            probs_0, probs_1, probs_2, probs_3 = softmax_four(C0.abs(), C1.abs(), C2.abs(), C3.abs())
            w0[mask_use_all] = probs_0[mask_use_all]
            w1[mask_use_all] = probs_1[mask_use_all]
            w2[mask_use_all] = probs_2[mask_use_all]
            w3[mask_use_all] = probs_3[mask_use_all]


        agree_012 = (s0 == s1) & (s1 == s2) & (s0 != s3)
        if agree_012.any():
            p0, p1, p2 = softmax_three(C0.abs(), C1.abs(), C2.abs())
            w0[agree_012] = p0[agree_012]
            w1[agree_012] = p1[agree_012]
            w2[agree_012] = p2[agree_012]

        agree_013 = (s0 == s1) & (s1 == s3) & (s0 != s2)
        if agree_013.any():
            p0, p1, p3 = softmax_three(C0.abs(), C1.abs(), C3.abs())
            w0[agree_013] = p0[agree_013]
            w1[agree_013] = p1[agree_013]
            w3[agree_013] = p3[agree_013]

        agree_023 = (s0 == s2) & (s2 == s3) & (s0 != s1)
        if agree_023.any():
            p0, p2, p3 = softmax_three(C0.abs(), C2.abs(), C3.abs())
            w0[agree_023] = p0[agree_023]
            w2[agree_023] = p2[agree_023]
            w3[agree_023] = p3[agree_023]

        agree_123 = (s1 == s2) & (s2 == s3) & (s1 != s0)
        if agree_123.any():
            p1, p2, p3 = softmax_three(C1.abs(), C2.abs(), C3.abs())
            w1[agree_123] = p1[agree_123]
            w2[agree_123] = p2[agree_123]
            w3[agree_123] = p3[agree_123]

        merged = W_base + w0 * C0 + w1 * C1 + w2 * C2 + w3 * C3
        
        merged_state[key] = merged.to(DTYPE)

# -------- save --------
print(f"Saving merged Llama-3.1 model to {SAVE_PATH}...")


model_base.load_state_dict(merged_state)
model_base.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"✅ Merged Llama-3.1 model saved successfully!")
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
