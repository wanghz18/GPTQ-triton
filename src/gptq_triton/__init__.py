import itertools
import json
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from . import fused_mlp, quant_linear
from .fused_attention import QuantLlamaAttention, make_quant_attn
from .fused_mlp import QuantLlamaMLP, make_fused_mlp
from .quant_linear import QuantLinear, make_quant, triton_matmul4
from .fused_norm import make_fused_norm
from IPython import embed


def make_fused_embedding(model, save_data=None):
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Embedding):
            continue

        mylayer = myEmbedding(m, name, save_data=save_data)
        parent_name = name.rsplit('.', 1)[0]
        parent = model.get_submodule(parent_name)

        # print(f"Replacing {name} with embedding; parent: {parent_name}, child's name: {name[len(parent_name) + 1:]}")

        setattr(parent, name[len(parent_name) + 1:], mylayer)

def make_fused_lm_header(model, save_data=None):
    for name, m in model.named_modules():
        if name == 'lm_head':
            mylayer = myLinear(m, name, save_data=save_data)
            parent_name = name.rsplit('.', 1)[0]
            parent = model.get_submodule(parent_name)

            model.lm_head = mylayer


class myEmbedding(torch.nn.Module):
    def __init__(
        self,
        m,
        name,
        save_data=None,
    ):
        super().__init__()

        self.layer = m
        self.name = name
        self.save_data = save_data

    def forward(self, x):
        save = False
        name = self.name

        save = True & (self.save_data is not None)
        if not save:
            return self.layer(x)
        
        self.save_data.add_data(f"{name}_input",
                                x,
                                "bsz, seqlen")
        y = self.layer(x)
        self.save_data.add_data(f"{name}_output",
                                y,
                                "bsz, seqlen, dim")
        weight = self.layer.weight
        self.save_data.add_data(f"{name}_weight",
                                weight,
                                "vocal_size, dim")
        return y

class myLinear(torch.nn.Module):
    def __init__(
        self,
        m,
        name,
        save_data=None,
    ):
        super().__init__()

        self.layer = m
        self.name = name
        self.save_data = save_data

    def forward(self, x):
        save = False
        name = self.name

        save = True & (self.save_data is not None)
        if not save:
            return self.layer(x)
        
        self.save_data.add_data(f"{name}_input",
                                x,
                                "bsz, seqlen, dim")
        y = self.layer(x)
        self.save_data.add_data(f"{name}_output",
                                y,
                                "bsz, seqlen, vocal_size")
        weight = self.layer.weight
        self.save_data.add_data(f"{name}_weight",
                                weight,
                                "vocal_size, dim")
        return y
  

def load_quant(checkpoint: str, warmup_autotune: bool = True, device: Optional[str] = 'cuda', fuse_mlp: Optional[bool] = None, save_data=None):
    """
    Load a quantized model from a checkpoint.
    Args:
        checkpoint: Path to the checkpoint directory.
        warmup_autotune: If True, run a warmup autotune pass. Otherwise autotune will run during forward passes.
        device: Device to run the model on; needed if warmup_autotune is True.
        fuse_mlp: If True, replace the MLP layers with fused versions.  If None, will apply fuse_mlp if the model's groupsize is -1, otherwise fuse_mlp will be disabled (it's slower when using grouping).
    Returns:
        The loaded model.
    """
    quant_config = json.load(open(Path(checkpoint) / 'quant_config.json'))
    wbits = quant_config['wbits']
    groupsize = quant_config['groupsize']

    # Load the model config
    config = LlamaConfig.from_pretrained(checkpoint)
    def noop(*args, **kwargs):
        pass
    # NOTE: This is a nasty hack, but it speeds up creation of the model by a huge amount.
    old_init = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    # Build the model
    # TODO: Is this needed?
    torch.set_default_dtype(torch.half)
    old_init_weights = transformers.modeling_utils._init_weights
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)

    # Restore the original init functions
    (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_) = old_init
    transformers.modeling_utils._init_weights = old_init_weights

    # Swap out linear layers for quantized ones
    make_quant(model, wbits, groupsize)

    # Load the quantized checkpoint
    print('Loading model ...')
    if (Path(checkpoint) / 'model.safetensors').exists():
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(Path(checkpoint) / 'model.safetensors'), strict=False)
    elif (Path(checkpoint) / 'model.pt').exists():
        model.load_state_dict(torch.load(Path(checkpoint) / 'model.pt'), strict=False)
    else:
        raise FileNotFoundError(f"Could not find model checkpoint at {checkpoint}; please ensure that the path is correct and contains a `model.pt` or `model.safetensors` file.")

    # Go through all the QuantLinear layers and if their bias is all zeros, set it to None
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear):
            if m.bias is not None and (m.bias == 0).all():
                m.bias = None
                #print(f"Removed bias from {name}")
    
    make_quant_attn(model, save_data)

    if fuse_mlp == True or (fuse_mlp is None and groupsize == -1):
        make_fused_mlp(model, save_data=save_data)
    
    make_fused_norm(model, save_data=save_data)
    make_fused_embedding(model, save_data=save_data)
    make_fused_lm_header(model, save_data=save_data)

    # Move the model to the correct device
    if device is not None:
        model = model.to(device)
    
    # Warm up the autotune cache
    if warmup_autotune:
        if device is None:
            raise ValueError("You must specify a device when warmup_autotune is True.")
        
        autotune_warmup(model)
    
    model.seqlen = 2048
    print('Done.')

    return model


def autotune_warmup(model):
    """
    The Triton kernels autotune themselves for specific input sizes.  But this takes time.
    This function collects information on all possible input sizes for the different kernels
    and then runs them through the autotuner.
    The intended use is to run this on startup so the autotuner doesn't have to run during
    actual inference.
    """
    from tqdm import tqdm

    warmups = itertools.chain(quant_linear.autotune_warmup(model), fused_mlp.autotune_warmup(model))
    warmups = list(warmups)

    print('Warming up autotune cache ...')
    with torch.no_grad():
        for m in tqdm(range(0, 12)):
            m = 2 ** m
            for func in warmups:
                func(m)
    return

      