from get_action import *
from transformers import AutoProcessor, AutoModelForImageTextToText,AutoModelForCausalLM, AutoTokenizer,Qwen2VLForConditionalGeneration
import torch
from eval_single import single_eval
import time
from tqdm import tqdm
from transfer import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", required=True)
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

JSON_PATH = args.json_path
MODEL_PATH = args.model_path


# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     MODEL_PATH,
#     device_map="auto",
#     trust_remote_code=True,
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2"
# )

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype="bfloat16", 
    device_map="auto",
    attn_implementation="sdpa"
)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True,use_fast=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data_list = json.load(f)

print(f"Loaded {len(data_list)} items from JSON.")

all_actions = []

model_name = "AC"
json_name = "AC_test"
ts = time.strftime("%Y%m%d_%H%M%S")
log_path = f"{model_name}_{json_name}_eval_log_{ts}.txt"

logs = []
total_type = 0
total_SR = 0
count = 0

prev_task = None
traj_SR_list = []
success_traj = 0
total_traj = 0

for idx, obs in enumerate(tqdm(data_list, desc="Evaluating", ncols=100)):
    print(f"\n=== Processing item {idx} ===")

    # try:
    #     action = get_action_atlas(model, processor, obs)
    # except Exception as e:
    #     action = "error"

    try:
        action, token = get_action_qwen3vl(model, processor, obs)
    except Exception as e:
        action = "error"
        token = 0
    
    try:
        action = transfer_qwen3vl2atlas(action)
    except Exception as e:
        action = "error"
    
    label = obs["action"]


    _type, SR = single_eval(action, label)

    total_type += _type
    total_SR += SR
    count += 1

    # 4. 输出日志
    print(f"[Step {idx}]")
    print(f"  action = {action}")
    print(f"  label  = {label}")
    print(f"  type   = {_type}")
    print(f"  SR     = {SR}")

    logs.append(
        f"Step {idx}\naction: {action}\nlabel: {label}\ntype: {_type}, SR: {SR}\n"
    )

    cur_task = obs["task"]
    task_changed = (prev_task is not None and cur_task != prev_task)
    is_last_item = (idx == len(data_list) - 1)

    traj_SR_list.append(SR)

    if task_changed or is_last_item:
        total_traj += 1
        if all(x == 1 for x in traj_SR_list):
            success_traj += 1
        traj_SR_list = []

    prev_task = cur_task
    # with open(log_path, "w", encoding="utf-8") as f:
    #     for line in logs:
    #         f.write(line + "\n")


avg_type = total_type / count if count > 0 else 0
avg_SR = total_SR / count if count > 0 else 0
TSR = success_traj / total_traj if total_traj > 0 else 0  # ⭐ TSR in [0,1]

print("\n=== Final Results ===")
print(f"Avg Type Accuracy: {avg_type:.4f}")
print(f"Avg SR Accuracy:   {avg_SR:.4f}")
print(f"TSR:               {TSR:.4f}  (0~1 range)")


print(f"\nLog saved to: {log_path}")

