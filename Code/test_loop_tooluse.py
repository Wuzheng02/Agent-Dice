import json
import re
import logging
import argparse
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from get_tool import get_tool_qwen3


FUNC_PATTERN = re.compile(r"(\w+)\s*\((.*?)\)", re.S)


def parse_tool_string(tool_str):
    results = defaultdict(dict)
    if not tool_str:
        return results

    for func_name, args_str in FUNC_PATTERN.findall(tool_str):
        if args_str.strip() == "":
            continue
        for arg in re.split(r",(?![^\(]*\))", args_str):
            if "=" not in arg:
                continue
            k, v = arg.split("=", 1)
            results[func_name][k.strip()] = v.strip()
    return results


def compute_metrics(pred, label):
    func_correct = 0
    func_label_cnt = len(label)

    param_name_correct = 0
    param_label_cnt = 0
    param_value_correct = 0

    for fname, label_params in label.items():
        param_label_cnt += len(label_params)

        if fname in pred:
            func_correct += 1
            pred_params = pred[fname]
            for p in label_params:
                if p in pred_params:
                    param_name_correct += 1
                    if pred_params[p] == label_params[p]:
                        param_value_correct += 1

    return {
        "func_correct": func_correct,
        "func_label_cnt": func_label_cnt,
        "param_name_correct": param_name_correct,
        "param_label_cnt": param_label_cnt,
        "param_value_correct": param_value_correct,
    }


def main(args):
    model_name = args.model
    test_json = args.test_json
    data_json = args.data_json
    gpu_id = args.gpu

    model_tag = model_name.rstrip("/").split("/")[-1]
    json_tag = test_json.rstrip("/").split("/")[-1].replace(".json", "")

    log_file = f"eval_{model_tag}__{json_tag}__gpu{gpu_id}.log"

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger()

    logger.info(f"MODEL: {model_name}")
    logger.info(f"TEST_JSON: {test_json}")
    logger.info(f"GPU: {gpu_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )


    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(data_json, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    total_metrics = defaultdict(int)

    for item in tqdm(test_data, desc=f"{model_tag} | {json_tag}"):
        idx = item["id"]
        sample = full_data[idx]

        system = sample.get("system", "")
        user = sample["conversations"][0]["value"]
        label_str = sample["conversations"][1]["value"]

        prompt = system + user

        try:
            pred_str = get_tool_qwen3(tokenizer, model, prompt)
        except Exception as e:
            logger.error(f"[ID {idx}] Inference failed: {e}")
            continue

        pred_parsed = parse_tool_string(pred_str)
        label_parsed = parse_tool_string(label_str)
        metrics = compute_metrics(pred_parsed, label_parsed)

        for k, v in metrics.items():
            total_metrics[k] += v

    func_acc = total_metrics["func_correct"] / max(1, total_metrics["func_label_cnt"])
    param_name_acc = total_metrics["param_name_correct"] / max(1, total_metrics["param_label_cnt"])
    param_value_acc = total_metrics["param_value_correct"] / max(1, total_metrics["param_label_cnt"])

    logger.info("FINAL METRICS")
    logger.info(f"Function Acc: {func_acc:.4f}")
    logger.info(f"Param Name Acc: {param_name_acc:.4f}")
    logger.info(f"Param Value Acc: {param_value_acc:.4f}")

    print(
        f"[DONE] GPU {gpu_id} | {model_tag} × {json_tag} | "
        f"Func {func_acc:.4f} | Param {param_name_acc:.4f} | Value {param_value_acc:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    main(args)
