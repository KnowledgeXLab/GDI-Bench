import torch
from transformers import AutoModelForCausalLM, TrainerCallback
import json
import argparse
import os
import shutil
import glob
import tqdm
import time
import random
import tqdm  

from torch.cuda.amp import autocast

import numpy as np
import matplotlib.pyplot as plt

def load_model(model_path, ref_model_path=None, device="cuda"):

    if ref_model_path is not None:
        copy_python_files(ref_model_path, model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    model.cpu().eval()
    return model

def compute_parameter_difference(base_model, target_model):
    diff_dict = {}
    base_state = dict(base_model.named_parameters())
    for name, param in target_model.named_parameters():
        if name in base_state:
            diff = (param - base_state[name]).abs()
            diff_dict[name] = diff.to(torch.float64)
    return diff_dict

def flatten_layer(layer_params):
    offsets = []
    current = 0
    tensors = []
    for name, tensor in layer_params.items():
        if "norm" in name:
            continue
        flat = tensor.flatten()
        tensors.append(flat)
        offsets.append( (name, current, current+flat.numel()) )
        current += flat.numel()
    return torch.cat(tensors), offsets  

def select_top_parameters(layer_tensor_dict, count_dict):
    param_elements = []
    
    for layer_name, k in count_dict.items():
        layer_params = layer_tensor_dict[layer_name]

        merged_tensor, index_map = flatten_layer(layer_params)

        values, indices = torch.topk(merged_tensor, min(k, merged_tensor.numel()))

        for val, idx in zip(values.tolist(), indices.tolist()):
            for (name, start, end) in index_map:
                if start <= idx < end:
                    param_idx = idx - start
                    param_elements.append([name, param_idx, val])
                    break

        del merged_tensor, index_map
        del values, indices
        torch.cuda.empty_cache()
                    
    return param_elements

def strategy_layer(diff_dicts, k):

    combined_diff = {}
    for key, diff_dict in diff_dicts.items():
        for param_name, tensor in tqdm.tqdm(diff_dict.items(), total=len(diff_dict)):
            parts = param_name.split(".")
            layer_key = ""
            for i, part in enumerate(parts):
                if part == "layers" and i+1 < len(parts) and parts[i+1].isdigit():
                    layer_idx = parts[i+1]
                    layer_key = f"{'_'.join(parts[:i])}_l{layer_idx}"
                    break
            
            if not layer_key:
                layer_key = parts[0]+parts[1]

            if layer_key not in combined_diff:
                combined_diff[layer_key] = {}
            
            if param_name not in combined_diff[layer_key]:
                combined_diff[layer_key][param_name] = tensor.clone()
            else:
                combined_diff[layer_key][param_name] += tensor
    
    layer_means = {}
    for layer_name, param_dict in combined_diff.items():
        sum_total = 0.0
        count = 0
        for param_name, tensor in param_dict.items():
            if "norm" in param_name:
                continue
            flat_tensor = tensor.detach().view(-1)
            sum_total += flat_tensor.sum().item()

        layer_means[layer_name] = sum_total 
    total = sum(layer_means.values())

    num_dict =  {key: int((value / total) * k) for key, value in layer_means.items()}

    param_elements = select_top_parameters(combined_diff, num_dict)

    ret = {}
    topk = param_elements[:k]
    topk_dict = {}
    for item in topk:
        pname, idx, val = item
        topk_dict.setdefault(pname, []).append(idx)
    ret[k] = topk_dict

    return ret, param_elements

diff_dicts = {}
def get_sorted_parameters(
    base_model_path, 
    expert_model_path, 
    k_per_model=None, 
    k=1000, 
    ref_model_path=None, 
    device="cuda"
):

    base_model = load_model(base_model_path, ref_model_path=ref_model_path, device=device)
    global diff_dicts

    expert_model = load_model(expert_model_path, ref_model_path=ref_model_path, device=device)
    diff = compute_parameter_difference(base_model, expert_model)
    diff_dicts[expert_model_path] = diff
    expert_model.to("cpu")
    del expert_model
    torch.cuda.empty_cache()
    del base_model

    sorted_params, tot_list = strategy_layer(diff_dicts, k)

    return sorted_params

def save_sorted_to_file(sorted_params, filename):
    with open(filename, 'w') as f:
        json.dump(sorted_params, f, indent=4)
def load_sorted_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def compute_and_save(base_model_path, expert_model_path, output_dir, k, k_per_model=None, ref_model_path=None):
    
    sorted_params = get_sorted_parameters(
        base_model_path,
        expert_model_path,
        k_per_model,
        k,
        ref_model_path=ref_model_path,
        device="cuda" 
    )
    for key, value in sorted_params.items():
        output_path = f"{output_dir}/sorted_params_strategy_layer_test_1_{key}.json"
        save_sorted_to_file(value, output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model Processing Tool")
    
    # Add command-line arguments
    parser.add_argument("--ref_model_path", required=True, help="Path to pretrained model")
    parser.add_argument("--expert_model_path", required=True, help="Path to expert model")
    parser.add_argument("--k", type=int, required=True, help=" k values")
    parser.add_argument("--output_dir", default="freeze_dict_layer", help="Output directory")
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    compute_and_save(args.ref_model_path, args.expert_model_path, args.output_dir, args.k)

