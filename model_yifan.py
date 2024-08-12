# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     ParallelEmbedding,
#     RowParallelLinear,
# )
from torch import nn
import numpy as np
import os
import tqdm
import re


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


layer_num = 0
layer_num_need = (0, 31)
inference_mode = "decoding"  # "decoding" or "prefill"
save_path = "./data_for_gem5_v2"


class SaveData():
    def __init__(self):
        self.description_dict = {}
        self.data_dict = {}
        self.shape_dict = {}
        self.shape_description_dict = {}
        self.source_code_dict = {}
        self.stride_dict = {}
        self.name_list = []
        self.count_number = -1
    
    def set_counter(self, num):
        self.count_number = num

    def add_data(self, name: str, data: torch.tensor, shape_description: str):
        # , source_code: str, description: str):
        num = name.split('_')[-1]
        if num.isdigit():
            if int(num) not in layer_num_need:
                return None
        
        if self.count_number != -1:
            name = f'{name}_partition_{self.count_number}'

        if 0 <= layer_num <= 31 and layer_num not in layer_num_need:
            return None
        # if not name.startswith("weight"):
        #     name += ("_" + inference_mode)
        if name.startswith("KVCache") and inference_mode == "prefill":
            return None
        if name in self.data_dict:
            return None
        self.name_list.append(name)
        self.data_dict[name] = data.to("cpu")
        # self.description_dict[name] = description
        self.shape_dict[name] = str(tuple(data.shape))
        self.shape_description_dict[name] = shape_description
        # self.source_code_dict[name] = source_code
        self.stride_dict[name] = str(data.stride())
        print(name, self.shape_dict[name])
        if not data.is_contiguous():
            print("not contiguous: ", name)
            print(shape_description)
            print(tuple(data.shape))
            print(data.stride())
            self.data_dict[name] = self.data_dict[name].contiguous()
        return None

    def get_data_by_name(self, name: str):
        if not name.startswith("weight"):
            name += ("_" + inference_mode)
        return self.data_dict[name]

    def save_to_c_file(self, name: str):
        path = save_path
        if not os.path.exists(path):
            os.makedirs(path)
        description = self.description_dict[name]
        shape_description = f"({self.shape_description_dict[name]})"
        shape = self.shape_dict[name]
        source_code = self.source_code_dict[name]
        stride = self.stride_dict[name]
        data = self.data_dict[name]
        data_size = torch.numel(data)

        data = data.numpy()
        np.save(f'{path}/{name}.npy', data)
        return None

        if name.startswith('activation_token_embedding'):
            # int64
            data_shape = tuple(data.shape)
            data = data.view(-1, 16).numpy().astype(np.int64)
            data = np.char.mod("%#010x", data)
            data = ", \n".join([", ".join(row) for row in data])
            if not os.path.exists(f"{path}/{name}_uint32.dat"):
                data = self.data_dict[name]
                data = data.view(-1, 16).numpy().astype(np.uint32)
                data.tofile(f"{path}/{name}_uint32.dat")
            return None
        if name.startswith("weight") or name.startswith('zeros'):
            # int32
            data_shape = tuple(data.shape)
            data = data.view(-1, 16).numpy().astype(np.uint32)
            data = np.char.mod("%#010x", data)
            data = ", \n".join([", ".join(row) for row in data])
            if not os.path.exists(f"{path}/{name}_uint32.dat"):
                data = self.data_dict[name]
                data = data.view(-1, 16).numpy().astype(np.uint32)
                data.tofile(f"{path}/{name}_uint32.dat")
            return None
        if True:
            if not os.path.exists(f"{path}/{name}_fp16.dat"):
                data_fp16_dat = data.view(-1, 16).to(torch.float16).numpy().view(np.uint16)
                data_fp16_dat.tofile(f"{path}/{name}_fp16.dat")
            # save data to c file fp32
        # save data to c file blockwise
        blockwise_size = 32 * 32
        if len(data.shape) >= 2:
            x, y = data.shape[-2:]
            padding_right = y % 32
            padding_right = 0 if padding_right == 0 else 32 - padding_right
            padding_bottom = x % 32
            padding_bottom = 0 if padding_bottom == 0 else 32 - padding_bottom
            data_blockwise = F.pad(data, (0, padding_right, 0, padding_bottom), mode='constant', value=0)
            data_blockwise_shape = data.shape[:-2] + (
            data_blockwise.shape[-2] // 32, 32, data_blockwise.shape[-1] // 32, 32)
            data_blockwise = data_blockwise.view(*data_blockwise_shape)
            # 把形状改为data.shape[:-2] + (data_blockwise.shape[-2] // 32, data_blockwise.shape[-1] // 32, 32, 32)
            data_blockwise = data_blockwise.transpose(-3, -2).contiguous()
            data_blockwise = data_blockwise.view(*data_blockwise.shape[:-2], 32 * 32)
            shape_blockwise = str(tuple(data_blockwise.shape))
            stride_blockwise = str(data_blockwise.stride())
            shape_description_blockwise = [s.strip() for s in self.shape_description_dict[name].split(",")]
            shape_description_blockwise[-2] = f"ceil({shape_description_blockwise[-2]} // 32)"
            shape_description_blockwise[-1] = f"ceil({shape_description_blockwise[-1]} // 32)"
            shape_description_blockwise.append("32 * 32")
            shape_description_blockwise = ", ".join(shape_description_blockwise)
            shape_description_blockwise = f"({shape_description_blockwise})"
            # save data to dat file blockwise fp16 bin
            if not os.path.exists(f"{path}/{name}_fp16_blockwise.dat"):
                data_fp16_blockwise = data_blockwise.view(-1, 16).to(torch.float16).numpy().view(np.uint16)
                data_fp16_blockwise.tofile(f"{path}/{name}_fp16_blockwise.dat")
        if name.startswith("weight_attention_w"):
            data_blockwise_T = data_blockwise.transpose(0, 1).contiguous()
            shape_blockwise_T = str(tuple(data_blockwise_T.shape))
            stride_blockwise_T = str(data_blockwise_T.stride())
            shape_description_blockwise = [s.strip() for s in self.shape_description_dict[name].split(",")]
            shape_description_blockwise_T = []
            shape_description_blockwise_T.append(f"ceil({shape_description_blockwise[-1]} // 32)")
            shape_description_blockwise_T.append(f"ceil({shape_description_blockwise[-2]} // 32)")
            shape_description_blockwise_T.append("32 * 32")
            shape_description_blockwise_T = ", ".join(shape_description_blockwise_T)
            shape_description_blockwise_T = f"({shape_description_blockwise_T})"
            # save data to dat file blockwise fp16 bin
            if not os.path.exists(f"{path}/{name}_fp16_blockwise_T.dat"):
                data_fp16_blockwise_T = data_blockwise_T.view(-1, 16).to(torch.float16).numpy().view(np.uint16)
                data_fp16_blockwise_T.tofile(f"{path}/{name}_fp16_blockwise_T.dat")
        if (((name.startswith("activation_ffn") and not name.startswith("activation_ffn_norm"))
                or re.search("activation_attention_x._before_reshape", name)
                or name.startswith("activation_attention_input") or name.startswith("activation_attention_output_after")
                or (name.startswith("activation_ffn") and torch.numel(data) >= 32 * 128)
                or name.startswith("activation_transformerblock_output"))
                and name.endswith("decoding") and data.shape[-2] == 1):
            data_blockwise_batching_shape = data.shape[:-3] + (
                data.shape[-3] // 32, 32, 1, data.shape[-1] // 32, 32)
            data_blockwise_batching = data.view(*data_blockwise_batching_shape)
            data_blockwise_batching = data_blockwise_batching.transpose(-4, -3).transpose(-3, -2).contiguous()
            data_blockwise_batching = data_blockwise_batching.view(*data_blockwise_batching.shape[:-2], 32 * 32)
            shape_blockwise_batching = str(tuple(data_blockwise_batching.shape))
            stride_blockwise_batching = str(data_blockwise_batching.stride())
            shape_description_blockwise_batching = [s.strip() for s in self.shape_description_dict[name].split(",")]
            shape_description_blockwise_batching[-3] = f"ceil({shape_description_blockwise_batching[-3]} // 32)"
            shape_description_blockwise_batching[-1] = f"ceil({shape_description_blockwise_batching[-1]} // 32)"
            shape_description_blockwise_batching.append("32 * 32")
            shape_description_blockwise_batching = ", ".join(shape_description_blockwise_batching)
            shape_description_blockwise_batching = f"({shape_description_blockwise_batching})"
            # save data to dat file blockwise fp16 bin
            if not os.path.exists(f"{path}/{name}_fp16_blockwise_batching.dat"):
                data_fp16_blockwise_batching = data_blockwise_batching.view(-1, 16).to(torch.float16).numpy().view(np.uint16)
                data_fp16_blockwise_batching.tofile(f"{path}/{name}_fp16_blockwise_batching.dat")
        return None


save_data = SaveData()


class RMSNorm_Before_Attention(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        x_before_rsqrt = x.pow(2).mean(-1, keepdim=True) + self.eps
        save_data.add_data(f"activation_attention_norm_x_before_rsqrt_layer_{layer_num}",
                           x_before_rsqrt,
                           "bsz, seqlen, 1",
                           "x_before_rsqrt: x_before_rsqrt = x.pow(2).mean(-1, keepdim=True) + self.eps",
                           "x_before_rsqrt: x_before_rsqrt=x^2.mean(-1)+eps")
        x_after_rsqrt = torch.rsqrt(x_before_rsqrt)
        save_data.add_data(f"activation_attention_norm_x_after_rsqrt_layer_{layer_num}",
                           x_after_rsqrt,
                           "bsz, seqlen, 1",
                           "x_after_rsqrt: x_after_rsqrt = torch.rsqrt(x_before_rsqrt)",
                           "x_after_rsqrt: x_after_rsqrt=rsqrt(x_before_rsqrt)")
        result = x * x_after_rsqrt
        return result

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        save_data.add_data(f"activation_attention_norm_input_layer_{layer_num}",
                           x,
                           "bsz, seqlen, dim",
                           "x: x_norm = self.attention_norm(x)",
                           "x: x_norm=RMSNorm(x)")
        mid_result = self._norm(x.float()).type_as(x)
        save_data.add_data(f"activation_attention_norm_mid_result_layer_{layer_num}",
                           mid_result,
                           "bsz, seqlen, dim",
                           "result: result = x * x_after_rsqrt",
                           "result: result=x*x_after_rsqrt")
        save_data.add_data(f"weight_attention_norm_weight_layer_{layer_num}",
                           self.weight,
                           "dim",
                           "weight: output = mid_result * self.weight",
                           "weight: output=mid_result*weight")
        output = mid_result * self.weight
        save_data.add_data(f"activation_attention_input_layer_{layer_num}",
                           output,
                           "bsz, seqlen, dim",
                           "x_norm: x_norm = self.attention_norm(x)",
                           "x_norm: x_norm=RMSNorm(x)")
        return output


class RMSNorm_Before_FFN(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        x_before_rsqrt = x.pow(2).mean(-1, keepdim=True) + self.eps
        save_data.add_data(f"activation_ffn_norm_x_before_rsqrt_layer_{layer_num}",
                           x_before_rsqrt,
                           "bsz, seqlen, 1",
                           "x_before_rsqrt: x_before_rsqrt = x.pow(2).mean(-1, keepdim=True) + self.eps",
                           "x_before_rsqrt: x_before_rsqrt=x^2.mean(-1)+eps")
        x_after_rsqrt = torch.rsqrt(x_before_rsqrt)
        save_data.add_data(f"activation_ffn_norm_x_after_rsqrt_layer_{layer_num}",
                           x_after_rsqrt,
                           "bsz, seqlen, 1",
                           "x_after_rsqrt: x_after_rsqrt = torch.rsqrt(x_before_rsqrt)",
                           "x_after_rsqrt: x_after_rsqrt=rsqrt(x_before_rsqrt)")
        result = x * x_after_rsqrt
        return result

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        save_data.add_data(f"activation_ffn_norm_input_layer_{layer_num}",
                           x,
                           "bsz, seqlen, dim",
                           "x: x_norm = self.ffn_norm(x)",
                           "x: x_norm=RMSNorm(x)")
        mid_result = self._norm(x.float()).type_as(x)
        save_data.add_data(f"activation_ffn_norm_mid_result_layer_{layer_num}",
                           mid_result,
                           "bsz, seqlen, dim",
                           "result: result = x * x_after_rsqrt",
                           "result: result=x*x_after_rsqrt")
        save_data.add_data(f"weight_ffn_norm_weight_layer_{layer_num}",
                           self.weight,
                           "dim",
                           "weight: output = mid_result * self.weight",
                           "weight: output=mid_result*weight")
        output = mid_result * self.weight
        save_data.add_data(f"activation_ffn_norm_output_layer_{layer_num}",
                           output,
                           "bsz, seqlen, dim",
                           "x_norm: x_norm = self.ffn_norm(x)",
                           "x_norm: x_norm=RMSNorm(x)")
        return output


class RMSNorm_Before_Logit(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        x_before_rsqrt = x.pow(2).mean(-1, keepdim=True) + self.eps
        save_data.add_data(f"activation_logit_norm_x_before_rsqrt",
                           x_before_rsqrt,
                           "bsz, seqlen, 1",
                           "x_before_rsqrt: x_before_rsqrt = x.pow(2).mean(-1, keepdim=True) + self.eps",
                           "x_before_rsqrt: x_before_rsqrt=x^2.mean(-1)+eps")
        x_after_rsqrt = torch.rsqrt(x_before_rsqrt)
        save_data.add_data(f"activation_logit_norm_x_after_rsqrt",
                           x_after_rsqrt,
                           "bsz, seqlen, 1",
                           "x_after_rsqrt: x_after_rsqrt = torch.rsqrt(x_before_rsqrt)",
                           "x_after_rsqrt: x_after_rsqrt=rsqrt(x_before_rsqrt)")
        result = x * x_after_rsqrt
        return result

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        save_data.add_data(f"activation_transformerblock_output_layer_{31}",
                           x,
                           "bsz, seqlen, dim",
                           "out: out = h + ffn_out",
                           "out: out=h+ffn_out")
        mid_result = self._norm(x.float()).type_as(x)
        save_data.add_data(f"activation_logit_norm_mid_result",
                           mid_result,
                           "bsz, seqlen, dim",
                           "result: result = x * x_after_rsqrt",
                           "result: result=x*x_after_rsqrt")
        save_data.add_data(f"weight_logit_norm_weight",
                           self.weight,
                           "dim",
                           "weight: output = mid_result * self.weight",
                           "weight: output=mid_result*weight")
        output = mid_result * self.weight
        save_data.add_data(f"activation_logit_output_before_outproj",
                           output,
                           "bsz, seqlen, dim",
                           "h_norm: h_norm = self.norm(h)",
                           "h_norm: h_norm=RMSNorm(h)")
        return output


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.




    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        if layer_num not in layer_num_need:
            self.cache_k = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()
        else:
            self.cache_k = torch.zeros(
                (
                    args.max_batch_size,
                    self.n_local_kv_heads,
                    args.max_seq_len,
                    self.head_dim,
                )
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    args.max_batch_size,
                    self.n_local_kv_heads,
                    args.max_seq_len,
                    self.head_dim,
                )
            ).cuda()

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        save_data.add_data(f"activation_attention_input_layer_{layer_num}",
                           x,
                           "bsz, seqlen, dim",
                           "x_norm: x_norm = self.attention_norm(x)",
                           "x_norm: x_norm=RMSNorm(x)")
        save_data.add_data(f"weight_attention_wq_layer_{layer_num}",
                           self.wq.get_master_weight().T,
                           "dim, n_heads * head_dim",
                           "wq: xq = self.wq(x)",
                           "wq: xq=x@wq")
        save_data.add_data(f"weight_attention_wk_layer_{layer_num}",
                           self.wk.get_master_weight().T,
                           "dim, n_kv_heads * head_dim",
                           "wk: xk = self.wk(x)",
                           "wk: xk=x@wk")
        save_data.add_data(f"weight_attention_wv_layer_{layer_num}",
                           self.wv.get_master_weight().T,
                           "dim, n_kv_heads * head_dim",
                           "wv: xv = self.wv(x)",
                           "wv: xv=x@wv")
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        save_data.add_data(f"activation_attention_xq_before_ROPE_layer_{layer_num}",
                           xq,
                           "bsz, seqlen, n_local_heads, head_dim",
                           "xq: xq = xq = self.wq(x); xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)",
                           "xq: xq=x@wq; reshape")
        save_data.add_data(f"activation_attention_xk_before_ROPE_layer_{layer_num}",
                           xk,
                           "bsz, seqlen, n_local_kv_heads, head_dim",
                           "xk: xk = xk = self.wk(x); xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)",
                           "xk: xk=x@wk; reshape")
        save_data.add_data(f"activation_attention_xv_layer_{layer_num}",
                           xv,
                           "bsz, seqlen, n_local_kv_heads, head_dim",
                           "xv: xv = xv = self.wv(x); xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)",
                           "xv: xv=x@wv; reshape")

        save_data.add_data(f"constant_attention_freqs_cis",
                           freqs_cis,
                           "seqlen, head_dim",
                           "freqs_cis: apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)",
                           "freqs_cis: apply_rotary_emb")
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        save_data.add_data(f"activation_attention_xq_after_ROPE_layer_{layer_num}",
                           xq,
                           "bsz, seqlen, n_local_heads, head_dim",
                           "xq: xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)",
                           "xq: xq, xk=apply_rotary_emb")
        save_data.add_data(f"activation_attention_xk_after_ROPE_layer_{layer_num}",
                           xk,
                           "bsz, seqlen, n_local_kv_heads, head_dim",
                           "xk: xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)",
                           "xk: xq, xk=apply_rotary_emb")

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        save_data.add_data(f"KVCache_k_layer_{layer_num}",
                           self.cache_k,
                           "max_batch_size, max_seq_len, n_local_kv_heads, head_dim",
                           "self.cache_k = self.cache_k.to(xq)",
                           "self.cache_k")
        save_data.add_data(f"KVCache_v_layer_{layer_num}",
                           self.cache_v,
                           "max_batch_size, max_seq_len, n_local_kv_heads, head_dim",
                           "self.cache_v = self.cache_v.to(xq)",
                           "self.cache_v")

        self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        save_data.add_data(f"activation_attention_keys_layer_{layer_num}",
                           keys,
                           "bsz, cache_len + seqlen, n_local_kv_heads, head_dim",
                           "keys: self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk; keys = self.cache_k[:bsz, : start_pos + seqlen]",
                           "keys: keys=contact(xk, cache_k)")
        save_data.add_data(f"activation_attention_values_layer_{layer_num}",
                           values,
                           "bsz, cache_len + seqlen, n_local_kv_heads, head_dim",
                           "values: self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv; values = self.cache_v[:bsz, : start_pos + seqlen]",
                           "values: values=contact(xv, cache_v)")

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        save_data.add_data(f"activation_attention_scores_before_mask_layer_{layer_num}",
                           scores,
                           "bsz, n_local_heads, seqlen, cache_len + seqlen",
                           "scores: xq = xq.transpose(1, 2); keys = keys.transpose(1, 2); scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)",
                           "scores: scores=xq@keys^T/sqrt(head_dim)")
        if mask is not None:
            save_data.add_data(f"constant_attention_mask",
                               mask,
                               "bsz, 1, seqlen, cache_len + seqlen",
                               "mask: mask = mask.unsqueeze(1)",
                               "mask: mask=unsqueeze(mask)")
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        save_data.add_data(f"activation_attention_scores_after_mask_layer_{layer_num}",
                           scores,
                           "bsz, n_local_heads, seqlen, cache_len + seqlen",
                           "scores: scores = scores + mask",
                           "scores: masked scores")
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        save_data.add_data(f"activation_attention_scores_after_softmax_layer_{layer_num}",
                           scores,
                           "bsz, n_local_heads, seqlen, cache_len + seqlen",
                           "scores: scores = F.softmax(scores.float(), dim=-1).type_as(xq)",
                           "scores: scores=softmax(scores)")
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        save_data.add_data(f"activation_attention_output_before_reshape_layer_{layer_num}",
                           output,
                           "bsz, n_local_heads, seqlen, head_dim",
                           "output: values = values.transpose(1, 2); output = torch.matmul(scores, values)",
                           "output: output=scores@values")
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        save_data.add_data(f"activation_attention_output_after_reshape_layer_{layer_num}",
                           output,
                           "bsz, seqlen, n_local_heads * head_dim",
                           "output: output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)",
                           "output: output=reshape(output)")

        save_data.add_data(f"weight_attention_wo_layer_{layer_num}",
                           self.wo.get_master_weight().T,
                           "n_heads * head_dim, dim",
                           "wo: output = self.wo(output)",
                           "wo: output=output@wo")
        output = self.wo(output)
        save_data.add_data(f"activation_attention_output_after_outproj_layer_{layer_num}",
                           output,
                           "bsz, seqlen, dim",
                           "output: output = self.wo(output)",
                           "output: output=output@wo")
        return output

    def forward_rearrangement(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module with rearrangement.

        Args:
            x:
            start_pos:
            freqs_cis:
            mask:

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, dim = x.shape
        cuda0 = x.device
        cuda1 = torch.device("cuda:2")
        x = x.reshape(bsz, seqlen, dim).to(cuda1)
        wq = self.wq.get_master_weight().to(cuda1).T.reshape(1, dim, self.n_local_heads, self.head_dim // 2, 2)
        wq = wq.transpose(3, 4).reshape(dim, self.n_local_heads * self.head_dim).contiguous()
        wk = self.wk.get_master_weight().to(cuda1).T.reshape(1, dim, self.n_local_kv_heads, self.head_dim // 2, 2)
        wk = wk.transpose(3, 4).reshape(dim, self.n_local_kv_heads * self.head_dim).contiguous()
        wv = self.wv.get_master_weight().to(cuda1).T.reshape(1, dim, self.n_local_kv_heads, self.head_dim // 2, 2)
        wv = wv.reshape(dim, self.n_local_kv_heads * self.head_dim).contiguous()

        save_data.add_data(f"activation_attention_input_layer_{layer_num}",
                           x.view(bsz, seqlen, dim),
                           "bsz, seqlen, dim",
                           "x_norm: x_norm = self.attention_norm(x)",
                           "x_norm: x_norm=RMSNorm(x)")
        save_data.add_data(f"weight_attention_wq_layer_{layer_num}",
                           wq,
                           "dim, n_heads * head_dim",
                           "wq: xq = torch.matmul(x, wq)",
                           "wq: xq=x@wq")
        save_data.add_data(f"weight_attention_wk_layer_{layer_num}",
                           wk,
                            "dim, n_kv_heads * head_dim",
                            "wk: xk = torch.matmul(x, wk)",
                            "wk: xk=x@wk")
        save_data.add_data(f"weight_attention_wv_layer_{layer_num}",
                           wv,
                            "dim, n_kv_heads * head_dim",
                            "wv: xv = torch.matmul(x, wv)",
                            "wv: xv=x@wv")

        xq = torch.matmul(x, wq)  # (bsz, seq_len, n_local_heads * head_dim)
        xk = torch.matmul(x, wk)  # (bsz, seq_len, n_local_kv_heads * head_dim)
        xv = torch.matmul(x, wv)  # (bsz, seq_len, n_local_kv_heads * head_dim)

        save_data.add_data(f"activation_attention_xq_before_reshape_layer_{layer_num}",
                           xq,
                           "bsz, seqlen, n_local_heads * head_dim",
                           "xq: xq = torch.matmul(x, wq)",
                           "xq: xq=x@wq")
        save_data.add_data(f"activation_attention_xk_before_reshape_layer_{layer_num}",
                           xk,
                           "bsz, seqlen, n_local_kv_heads * head_dim",
                           "xk: xk = torch.matmul(x, wk)",
                           "xk: xk=x@wk")
        save_data.add_data(f"activation_attention_xv_before_reshape_layer_{layer_num}",
                           xv,
                           "bsz, seqlen, n_local_kv_heads * head_dim",
                           "xv: xv = torch.matmul(x, wv)",
                           "xv: xv=x@wv")

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1,
                                                                               2).contiguous()  # (bsz, n_local_heads, seqlen, head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim).transpose(1,
                                                                                  2).contiguous()  # (bsz, n_local_kv_heads, seqlen, head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim).transpose(1,
                                                                                  2).contiguous()  # (bsz, n_local_kv_heads, seqlen, head_dim)

        save_data.add_data(f"activation_attention_xq_before_ROPE_layer_{layer_num}",
                           xq,
                           "bsz, n_heads, seqlen, head_dim",
                           "xq: xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)",
                           "xq: xq=reshape(xq)")
        save_data.add_data(f"activation_attention_xk_before_ROPE_layer_{layer_num}",
                           xk,
                           "bsz, n_kv_heads, seqlen, head_dim",
                           "xk: xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim).transpose(1, 2)",
                           "xk: xk=reshape(xk)")
        save_data.add_data(f"activation_attention_xv_layer_{layer_num}",
                           xv,
                           "bsz, n_kv_heads, seqlen, head_dim",
                           "xv: xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim).transpose(1, 2)",
                           "xv: xv=reshape(xv)")

        assert xq.shape == (bsz, self.n_local_heads, seqlen,
                            self.head_dim), f"xq shape is {xq.shape} while expected shape is {(bsz, self.n_local_heads, seqlen, self.head_dim)}"
        assert xk.shape == (bsz, self.n_local_kv_heads, seqlen,
                            self.head_dim), f"xk shape is {xk.shape} while expected shape is {(bsz, self.n_local_kv_heads, seqlen, self.head_dim)}"
        assert xv.shape == (bsz, self.n_local_kv_heads, seqlen,
                            self.head_dim), f"xv shape is {xv.shape} while expected shape is {(bsz, self.n_local_kv_heads, seqlen, self.head_dim)}"





        # save_data.add_data(f"activation_attention_xk_after_cache_layer_{layer_num}",
        #                     xk,
        #                     "bsz, n_kv_heads, cache_len + seqlen, head_dim",
        #                     "xk: self.cache_k[:, :, start_pos: start_pos + seqlen] = xk; xk = self.cache_k[:, :, : start_pos + seqlen]",
        #                     "xk: xk=contact(xk, cache_k)")
        # save_data.add_data(f"activation_attention_xv_after_cache_layer_{layer_num}",
        #                     xv,
        #                     "bsz, n_kv_heads, cache_len + seqlen, head_dim",
        #                     "xv: self.cache_v[:, :, start_pos: start_pos + seqlen] = xv; xv = self.cache_v[:, :, : start_pos + seqlen]",
        #                     "xv: xv=contact(xv, cache_v)")

        # rotary position embedding
        freqs_cis = torch.view_as_real(freqs_cis).to(cuda1)  # (seqlen, head_dim)
        freqs_cis = freqs_cis.reshape(seqlen, self.head_dim // 2, 2).permute(2, 0, 1)
        cos_pos = freqs_cis[0].contiguous()  # (seq_len, head_dim // 2)
        sin_pos = freqs_cis[1].contiguous()  # (seq_len, head_dim // 2)
        cos_pos = cos_pos.repeat(1, 2)  # (seq_len, head_dim)
        sin_pos = sin_pos.repeat(1, 2)  # (seq_len, head_dim)
        origin_xq = xq.float()  # (bsz, n_heads, seqlen, head_dim)
        origin_xk = xk.float()  # (bsz, n_kv_heads, seqlen, head_dim)
        rope_xq = torch.cat([-origin_xq[:, :, :, self.head_dim // 2:], origin_xq[:, :, :, :self.head_dim // 2]], dim=-1)
        rope_xk = torch.cat([-origin_xk[:, :, :, self.head_dim // 2:], origin_xk[:, :, :, :self.head_dim // 2]], dim=-1)

        save_data.add_data(f"constant_ROPE_cos_pos",
                           cos_pos,
                           "seqlen, head_dim",
                           "cos_pos: xq = origin_xq * cos_pos + rope_xq * sin_pos",
                           "cos_pos: xq=origin_xq*cos_pos+rope_xq*sin_pos")
        save_data.add_data(f"constant_ROPE_sin_pos",
                           sin_pos,
                           "seqlen, head_dim",
                           "sin_pos: xq = origin_xq * cos_pos + rope_xq * sin_pos",
                           "sin_pos: xq=origin_xq*cos_pos+rope_xq*sin_pos")
        save_data.add_data(f"activation_ROPE_origin_xq_layer_{layer_num}",
                           origin_xq,
                           "bsz, n_heads, seqlen, head_dim",
                           "origin_xq: origin_xq = xq.float()",
                           "origin_xq: xq=to_float16(xq)")
        save_data.add_data(f"activation_ROPE_origin_xk_layer_{layer_num}",
                           origin_xk,
                           "bsz, n_kv_heads, seqlen, head_dim",
                           "origin_xk: origin_xk = xk.float()",
                           "origin_xk: xk=to_float16(xk)")
        save_data.add_data(f"activation_ROPE_rope_xq_layer_{layer_num}",
                           rope_xq,
                           "bsz, n_heads, seqlen, head_dim",
                           "rope_xq: rope_xq = torch.cat([-origin_xq[:, :, :, self.head_dim // 2:], origin_xq[:, :, :, :self.head_dim // 2]], dim=-1)",
                           "rope_xq: rope_xq=concat(-origin_xq[:, :, :, head_dim // 2:], origin_xq[:, :, :, :head_dim // 2])")
        save_data.add_data(f"activation_ROPE_rope_xk_layer_{layer_num}",
                           rope_xk,
                           "bsz, n_kv_heads, seqlen, head_dim",
                           "rope_xk: rope_xk = torch.cat([-origin_xk[:, :, :, self.head_dim // 2:], origin_xk[:, :, :, :self.head_dim // 2]], dim=-1)",
                           "rope_xk: rope_xk=concat(-origin_xk[:, :, :, head_dim // 2:], origin_xk[:, :, :, :head_dim // 2])")

        xq = origin_xq * cos_pos + rope_xq * sin_pos
        xk = origin_xk * cos_pos + rope_xk * sin_pos

        save_data.add_data(f"activation_ROPE_result_xq_layer_{layer_num}",
                           xq,
                           "bsz, n_heads, seqlen, head_dim",
                           "xq: xq = origin_xq * cos_pos + rope_xq * sin_pos",
                           "xq: xq=origin_xq*cos_pos+rope_xq*sin_pos")
        save_data.add_data(f"activation_ROPE_result_xk_layer_{layer_num}",
                           xk,
                           "bsz, n_kv_heads, seqlen, head_dim",
                           "xk: xk = origin_xk * cos_pos + rope_xk * sin_pos",
                           "xk: xk=origin_xk*cos_pos+rope_xk*sin_pos")

        # assert xq.shape == (bsz, self.n_local_heads, seqlen, self.head_dim), f"xq shape is {xq.shape} while expected shape is {(bsz, self.n_local_heads, seqlen, self.head_dim)}"
        # assert xk.shape == (bsz, self.n_local_kv_heads, start_pos+ seqlen, self.head_dim), f"xk shape is {xk.shape} while expected shape is {(bsz, self.n_local_kv_heads, start_pos + seqlen, self.head_dim)}"

        xq = xq.type_as(x)
        xk = xk.type_as(x)

        save_data.add_data(f"activation_attention_xq_after_ROPE_layer_{layer_num}",
                           xq,
                           "bsz, n_heads, seqlen, head_dim",
                           "xq: xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)",
                           "xq: xq, xk=apply_rotary_emb")
        save_data.add_data(f"activation_attention_xk_after_ROPE_layer_{layer_num}",
                           xk,
                           "bsz, n_kv_heads, seqlen, head_dim",
                           "xk: xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)",
                           "xk: xq, xk=apply_rotary_emb")

        # kv-cache
        # cache_k = self.cache_k.to(cuda1) # (bsz, cache_len, n_kv_heads, head_dim)
        # cache_v = self.cache_v.to(cuda1) # (bsz, cache_len, n_kv_heads, head_dim)
        # cache_k = cache_k[:, :start_pos] # (bsz, start_pos, n_kv_heads, head_dim)
        # cache_v = cache_v[:, :start_pos] # (bsz, start_pos, n_kv_heads, head_dim)
        # cache_k = cache_k.transpose(1, 2) # (bsz, n_kv_heads, start_pos, head_dim)
        # cache_v = cache_v.transpose(1, 2) # (bsz, n_kv_heads, start_pos, head_dim)
        # cache_k = torch.concat([cache_k, xk], dim=2) # (bsz, n_kv_heads, cache_len, head_dim)
        # cache_v = torch.concat([cache_v, xv], dim=2) # (bsz, n_kv_heads, cache_len, head_dim)
        self.cache_k = self.cache_k.to(cuda1)  # (bsz, n_kv_heads, cache_len, head_dim)
        self.cache_v = self.cache_v.to(cuda1)  # (bsz, n_kv_heads, cache_len, head_dim)

        save_data.add_data(f"KVCache_k_layer_{layer_num}",
                           self.cache_k,
                           "max_batch_size, n_kv_heads, max_seq_len, head_dim",
                           "self.cache_k = self.cache_k.to(xq)",
                           "self.cache_k")
        save_data.add_data(f"KVCache_v_layer_{layer_num}",
                           self.cache_v,
                           "max_batch_size, n_kv_heads, max_seq_len, head_dim",
                           "self.cache_v = self.cache_v.to(xq)",
                           "self.cache_v")

        self.cache_k[:, :, start_pos: start_pos + seqlen] = xk
        self.cache_v[:, :, start_pos: start_pos + seqlen] = xv
        xk = self.cache_k[:, :, : start_pos + seqlen]  # (bsz, n_heads, cache_len, head_dim)
        xv = self.cache_v[:, :, : start_pos + seqlen]  # (bsz, n_heads, cache_len, head_dim)

        keys = repeat_kv(xk, self.n_rep)  # (bs, n_heads, seqlen, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, n_heads, seqlen, head_dim)

        save_data.add_data(f"activation_attention_keys_layer_{layer_num}",
                           keys,
                           "bsz, n_heads, cache_len, head_dim",
                           "keys: repeat_kv(self.cache_k[:, :, : start_pos + seqlen], self.n_rep)",
                           "keys: keys=repeat_kv(cache_k)")
        save_data.add_data(f"activation_attention_values_layer_{layer_num}",
                           values,
                           "bsz, n_heads, cache_len, head_dim",
                           "values: repeat_kv(self.cache_v[:, :, : start_pos + seqlen], self.n_rep)",
                           "values: values=repeat_kv(cache_v)")
        save_data.add_data(f"activation_attention_keys_T_layer_{layer_num}",
                           keys.transpose(2, 3).contiguous(),
                           "bsz, n_heads, head_dim, cache_len",
                           "keys: keys = keys.transpose(2, 3)",
                           "keys: keys=keys^T")

        scores = torch.matmul(xq, keys.transpose(2, 3))

        save_data.add_data(f"activation_attention_scores_before_div_sqrt_dim_layer_{layer_num}",
                            scores,
                            "bsz, n_heads, seqlen, cache_len",
                            "scores: scores = torch.matmul(xq, keys.transpose(2, 3))",
                            "scores: scores=xq@keys^T")

        scores = scores / math.sqrt(self.head_dim)

        save_data.add_data(f"activation_attention_scores_before_mask_layer_{layer_num}",
                           scores,
                           "bsz, n_local_heads, seqlen, cache_len + seqlen",
                           "scores: xq = xq.transpose(1, 2); keys = keys.transpose(1, 2); scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)",
                           "scores: scores=xq@keys^T/sqrt(head_dim)")

        if mask is not None:
            mask = mask.to(cuda1)
            save_data.add_data(f"constant_attention_mask",
                               mask,
                               "bsz, 1, seqlen, cache_len + seqlen",
                               "mask: mask = mask.unsqueeze(1)",
                               "mask: mask=unsqueeze(mask)")
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        save_data.add_data(f"activation_attention_scores_after_mask_layer_{layer_num}",
                           scores,
                           "bsz, n_local_heads, seqlen, cache_len + seqlen",
                           "scores: scores = scores + mask",
                           "scores: masked scores")

        # scores
        scores_float = scores.float()
        scores_sub_max = scores_float - torch.max(scores_float, dim=-1, keepdim=True)[0]
        scores_exp = torch.exp(scores_sub_max)
        scores_exp_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        save_data.add_data(f"activation_attention_scores_sub_max_layer_{layer_num}",
                            scores_sub_max,
                            "bsz, n_local_heads, seqlen, cache_len + seqlen",
                            "scores: scores_float = scores.float(); scores_sub_max = scores_float - torch.max(scores_float, dim=-1, keepdim=True)[0]",
                            "scores: scores_sub_max=scores_float-max(scores_float)")
        save_data.add_data(f"activation_attention_scores_exp_layer_{layer_num}",
                            scores_exp,
                            "bsz, n_local_heads, seqlen, cache_len + seqlen",
                            "scores: scores_exp = torch.exp(scores_sub_max)",
                            "scores: scores_exp=exp(scores_sub_max)")
        save_data.add_data(f"activation_attention_scores_exp_sum_layer_{layer_num}",
                            scores_exp_sum,
                            "bsz, n_local_heads, seqlen, 1",
                            "scores: scores_exp_sum = torch.sum(scores_exp, dim=-1, keepdim=True)",
                            "scores: scores_exp_sum=sum(scores_exp)")

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        assert scores.shape == (bsz, self.n_local_heads, seqlen, start_pos + seqlen), f"scores shape is {scores.shape} while expected shape is {(bsz, self.n_local_heads, seqlen, start_pos + seqlen)}"
        save_data.add_data(f"activation_attention_scores_after_softmax_layer_{layer_num}",
                           scores,
                           "bsz, n_local_heads, seqlen, cache_len + seqlen",
                           "scores: scores = F.softmax(scores.float(), dim=-1).type_as(xq)",
                           "scores: scores=softmax(scores)")
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        assert output.shape == (bsz, self.n_local_heads, seqlen, self.head_dim), f"output shape is {output.shape} while expected shape is {(bsz, self.n_local_heads, seqlen, self.head_dim)}"
        save_data.add_data(f"activation_attention_output_before_reshape_layer_{layer_num}",
                           output,
                           "bsz, n_local_heads, seqlen, head_dim",
                           "output: values = values.transpose(1, 2); output = torch.matmul(scores, values)",
                           "output: output=scores@values")
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        save_data.add_data(f"activation_attention_output_after_reshape_layer_{layer_num}",
                           output,
                           "bsz, seqlen, n_local_heads * head_dim",
                           "output: output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)",
                           "output: output=reshape(output)")

        save_data.add_data(f"weight_attention_wo_layer_{layer_num}",
                           self.wo.get_master_weight().T,
                           "n_heads * head_dim, dim",
                           "wo: output = self.wo(output)",
                           "wo: output=output@wo")
        output = self.wo.to(cuda1)(output)
        save_data.add_data(f"activation_attention_output_after_outproj_layer_{layer_num}",
                           output,
                           "bsz, seqlen, dim",
                           "output: output = self.wo(output)",
                           "output: output=output@wo")
        return output.to(cuda0)

    def compare_scores_and_values(self, scores: torch.Tensor, values: torch.Tensor):
        """
        Compare the scores and values tensors.

        Args:
            scores (torch.Tensor): Scores tensor.
            values (torch.Tensor): Values tensor.

        Raises:
            AssertionError: If the scores and values tensors don't have the same shape.
        """
        reference_scores = save_data.get_data_by_name(f"activation_attention_scores_before_mask_layer_{layer_num}").to(
            scores.device)
        reference_values = save_data.get_data_by_name(f"activation_attention_values_layer_{layer_num}").to(
            values.device).transpose(1, 2).contiguous()
        assert scores.shape == reference_scores.shape
        assert values.shape == reference_values.shape
        scores = scores.float()
        reference_scores = reference_scores.float()
        values = values.float()
        reference_values = reference_values.float()
        # 统计完全不差的有多少
        scores_diff = torch.abs(scores.view(torch.int32) - reference_scores.view(torch.int32))
        values_diff = torch.abs(values.view(torch.int32) - reference_values.view(torch.int32))
        print("scores_number: ", torch.numel(scores_diff), "not_zero_number: ", torch.count_nonzero(scores_diff).item())
        print("values_number: ", torch.numel(values_diff), "not_zero_number: ", torch.count_nonzero(values_diff).item())
        # scores = scores.view(-1)
        # reference_scores = reference_scores.view(-1)
        # condi = torch.abs(scores - reference_scores) > 1e-2 * torch.abs(reference_scores) + 1e-5
        # print("unqualified number: ", torch.stack([scores[condi], reference_scores[condi]], dim=1))


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        save_data.add_data(f"activation_ffn_input_layer_{layer_num}",
                           x,
                           "bsz, seqlen, dim",
                           "x: x_norm = self.ffn_norm(x)",
                           "x: x_norm=RMSNorm(x)")
        save_data.add_data(f"weight_ffn_w1_layer_{layer_num}",
                           self.w1.get_master_weight().T,
                           "dim, hidden_dim",
                           "w1: x1 = self.w1(x)",
                           "w1: x1=x@w1")
        x1_before_silu = self.w1(x)
        save_data.add_data(f"activation_ffn_x1_before_silu_layer_{layer_num}",
                           x1_before_silu,
                           "bsz, seqlen, hidden_dim",
                           "x1_before_silu: x1_before_silu = self.w1(x)",
                           "x1_before_silu: x1_before_silu=x@w1")
        x1_after_silu = F.silu(x1_before_silu)
        save_data.add_data(f"activation_ffn_x1_after_silu_layer_{layer_num}",
                           x1_after_silu,
                           "bsz, seqlen, hidden_dim",
                           "x1_after_silu: x1_after_silu = F.silu(x1_before_silu)",
                           "x1_after_silu: x1_after_silu=silu(x1_before_silu)")
        save_data.add_data(f"weight_ffn_w3_layer_{layer_num}",
                           self.w3.get_master_weight().T,
                           "dim, hidden_dim",
                           "w3: x3 = self.w3(x1_after_silu)",
                           "w3: x3=x1_after_silu@w3")
        x3 = self.w3(x)
        save_data.add_data(f"activation_ffn_x3_layer_{layer_num}",
                           x3,
                           "bsz, seqlen, hidden_dim",
                           "x3: x3 = self.w3(x)",
                           "x3: x3=x@w3")
        x2 = x1_after_silu * x3
        save_data.add_data(f"activation_ffn_x2_layer_{layer_num}",
                           x2,
                           "bsz, seqlen, hidden_dim",
                           "x2: x2 = x1_after_silu * x3",
                           "x2: x2=x1_after_silu*x3")
        save_data.add_data(f"weight_ffn_w2_layer_{layer_num}",
                           self.w2.get_master_weight().T,
                           "hidden_dim, dim",
                           "w2: output = self.w2(x2)",
                           "w2: output=x2@w2")
        output = self.w2(x2)
        save_data.add_data(f"activation_ffn_output_layer_{layer_num}",
                           output,
                           "bsz, seqlen, dim",
                           "output: output = self.w2(x2)",
                           "output: output=x2@w2")
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        global layer_num
        layer_num = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm_Before_Attention(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm_Before_FFN(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        save_data.add_data(f"activation_attention_norm_input_layer_{layer_num}",
                           x,
                           "bsz, seqlen, dim",
                           "x: x_norm = self.attention_norm(x)",
                           "x: x_norm=RMSNorm(x)")
        if layer_num not in layer_num_need:
            attention_output = self.attention(
                self.attention_norm(x), start_pos, freqs_cis, mask
            )
        else:
            attention_output = self.attention.forward_rearrangement(
                self.attention_norm(x), start_pos, freqs_cis, mask
            )
        save_data.add_data(f"activation_attention_output_after_outproj_layer_{layer_num}",
                           attention_output,
                           "bsz, seqlen, dim",
                           "output: output = self.wo(output)",
                           "output: output=output@wo")

        h = x + attention_output
        save_data.add_data(f"activation_ffn_norm_input_layer_{layer_num}",
                           h,
                           "bsz, seqlen, dim",
                           "h: h_norm = self.ffn_norm(h)",
                           "h: h_norm=RMSNorm(h)")
        ffn_out = self.feed_forward(self.ffn_norm(h))
        save_data.add_data(f"activation_ffn_output_layer_{layer_num}",
                           ffn_out,
                           "bsz, seqlen, dim",
                           "output: output = self.w2(x2)",
                           "output: output=x2@w2")
        out = h + ffn_out
        save_data.add_data(f"activation_transformerblock_output_layer_{layer_num}",
                           out,
                           "bsz, seqlen, dim",
                           "out: out = h + ffn_out",
                           "out: out=h+ffn_out")
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm_Before_Logit(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        global layer_num, inference_mode
        layer_num = 0
        inference_mode = "decoding" if start_pos > 0 else "prefill"

        _bsz, seqlen = tokens.shape

        save_data.add_data(f"activation_token_embedding_tokens",
                           tokens,
                           "bsz, seqlen",
                           "tokens: h = self.tok_embeddings(tokens)",
                           "tokens: h=tok_embeddings(tokens)")
        save_data.add_data(f"weight_token_embedding_tok_embeddings",
                           self.tok_embeddings.weight.data,
                           "vocab_size, dim",
                           "tok_embeddings: h = self.tok_embeddings(tokens)",
                           "tok_embeddings: h=tok_embeddings(tokens)")
        h = self.tok_embeddings(tokens)
        save_data.add_data(f"activation_attention_norm_input_layer_{0}",
                           h,
                           "bsz, seqlen, dim",
                           "x: x_norm = self.attention_norm(x)",
                           "x: x_norm=RMSNorm(x)")

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for i in range(10, 20):
            self.layers[i] = self.layers[i].to(torch.device("cuda:3"))

        for layer in self.layers:
            if h.device != layer.attention.wq.weight.device:
                h = h.to(layer.attention.wq.weight.device)
                freqs_cis = freqs_cis.to(layer.attention.wq.weight.device)
                mask = mask.to(layer.attention.wq.weight.device) if mask is not None else None
            h = layer(h, start_pos, freqs_cis, mask)
            layer_num += 1
        save_data.add_data(f"activation_transformerblock_output_layer_{31}",
                           h,
                           "bsz, seqlen, dim",
                           "out: out = h + ffn_out",
                           "out: out=h+ffn_out")
        h = self.norm(h)
        save_data.add_data(f"activation_logit_output_before_outproj",
                           h,
                           "bsz, seqlen, dim",
                           "h_norm: h_norm = self.norm(h)",
                           "h_norm: h_norm=RMSNorm(h)")
        save_data.add_data(f"weight_logit_output",
                           self.output.get_master_weight().T,
                           "dim, vocab_size",
                           "output: output = self.output(h)",
                           "output: output=h@output")
        output = self.output(h)
        save_data.add_data(f"activation_logit_output_after_outproj",
                           output,
                           "bsz, seqlen, vocab_size",
                           "output: output = self.output(h)",
                           "output: output=h@output")

        print("====================================")
        print("torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction: ", torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        print("torch.backends.cuda.matmul.allow_tf32: ", torch.backends.cuda.matmul.allow_tf32)
        print("====================================")
        return output

    def save_params(self):
        name_dict = save_data.shape_dict.keys()
        print(name_dict)
        print(save_data.description_dict)
        print(save_data.shape_description_dict)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, "data_name.txt"), "w") as f:
            for name in save_data.name_list:
                f.write(name + "\n")
        with open(os.path.join(save_path, "data_description.txt"), "w") as f:
            for name in save_data.name_list:
                f.write(f"{name}({save_data.shape_description_dict[name]}): {save_data.description_dict[name]}\n")
        for name in tqdm.tqdm(name_dict):
            save_data.save_to_c_file(name)
