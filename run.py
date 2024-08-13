#!/usr/bin/env python3
"""
Benchmarks the generation speed of a model.  While Benchmark.ipynb provides nice detailed performance data, it measures the kernels in isolation.
This script measures "real world" performance by running the whole model in generation mode.
It tests a grid of prompt lengths and generation lengths, and saves the timing results to `results.json`.
"""
import argparse
import itertools
import json
import os
import random
import time

import original_quant
import torch
import transformers
from src.gptq_triton import load_quant
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from IPython import embed
from model_yifan import SaveData
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to model, either a HuggingFace model or a quantized model')
parser.add_argument('--quant', action='store_true', help='Whether the model is quantized')
parser.add_argument('--cuda', type=str, help='Whether to use the old CUDA kernel and format; this must be set to the path to the CUDA quantized model, and --model must be set to a HF model')
parser.add_argument('--average', type=int, default=10, help='Number of times to run each test to get an average')
parser.add_argument('--seed', type=int, default=0, help='Seed')


def main():
    args = parser.parse_args()
    save_data = SaveData()

    if args.cuda:
        model = load_cuda_quant(args.model, args.cuda, 4, -1)
        model.eval()
        model.to('cuda')
    elif not args.quant:
        model = get_llama(args.model)
        model.eval()
        model.to('cuda')
    else:
        model = load_quant(args.model, fuse_mlp=True, save_data=save_data)
        model.eval()
        model.to('cuda')
    

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    random.seed(args.seed)

    encoded_prompt = torch.randint(32000, (32, 128), dtype=torch.int32).to('cuda')
    # tokenizer.encode("TODO", add_special_tokens=False, return_tensors='pt').to('cuda')
    
    # encoded_prompt = encoded_prompt.to('cuda')

    start_time = time.time()
    with torch.no_grad():
        y = model.forward(encoded_prompt)
    for it in save_data.name_list:
        data = save_data.data_dict[it].cpu().numpy()
        filename = f'save/{it}.npy'
        new_name = filename.replace('(', '_')
        new_name = new_name.replace(')', '_')
        new_name = new_name.replace('*', '_')
        new_name = new_name.replace('|', '_')
        new_name = new_name.replace('^', '_')
        filename = new_name
        print(filename, data.shape)
        np.save(filename, data)
    
    def cell(s):
        return f'<th class="tg-c3ow">{s}</th>'
    
    with open('table.html', 'w') as f:
        f.write('<style type="text/css"> \
.tg {border-collapse:collapse;border-spacing:0;} \
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px; \
overflow:hidden;padding:10px 5px;word-break:normal;} \
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px; \
font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;} \
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top} \
</style>')
        f.write('<table class="tg">\n')
        f.write('<thead><tr><th class="tg-c3ow">name</th><th class="tg-c3ow">shape</th><th class="tg-c3ow">shape</th><th class="tg-c3ow">type</th></tr></thead>\n')
        f.write('<tbody>\n')
        for it in save_data.name_list:
            tp = str(save_data.data_dict[it].dtype)
            sp = cell(f'({save_data.shape_description_dict[it]})')
            f.write(f'<tr>{cell(it)}{sp}{cell(save_data.shape_dict[it])}{cell(tp)}</tr>\n')
        f.write('</tbody></table>\n')
        



def get_llama(model: str):
    """
    Load a pretrained Llama model
    """
    def skip(*args, **kwargs):
        pass
    # NOTE: This is a nasty hack, but it speeds up model building by a huge amount
    old_inits = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048

    # Restore the old initializers
    torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = old_inits

    return model


def load_cuda_quant(model, checkpoint, wbits, groupsize):
    """
    Load a quantized model using the old CUDA kernel
    """
    config = LlamaConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    original_inits = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    original_init_weights = transformers.modeling_utils._init_weights
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)

    transformers.modeling_utils._init_weights = original_init_weights
    torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = original_inits

    model = model.eval()
    layers = original_quant.find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    original_quant.make_quant(model, layers, wbits, groupsize, faster=False)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model


if __name__ == '__main__':
    main()