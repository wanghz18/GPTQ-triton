import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from .quant_linear import QuantLinear
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, LlamaConfig
from IPython import embed


def make_quant_attn(model, save_data=None):
    """
    Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
    """
    for name, m in model.named_modules():
        if not isinstance(m, LlamaAttention):
            continue

        q_proj = m.q_proj
        k_proj = m.k_proj
        v_proj = m.v_proj

        qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
        qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
        scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)

        qkv_layer = QuantLinear(q_proj.bits, q_proj.groupsize, q_proj.infeatures, q_proj.outfeatures + k_proj.outfeatures + v_proj.outfeatures, bias=False)
        qkv_layer.qweight = qweights
        qkv_layer.qzeros = qzeros
        qkv_layer.scales = scales
        qkv_layer.bias = None

        attn = QuantLlamaAttention(m.config, qkv_layer, m.o_proj, m.rotary_emb, name, save_data)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        #print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, attn)


class QuantLlamaAttention(nn.Module):
    """
    Modified version of LlamaAttention that fuses the q, k, v projections.
    """

    def __init__(
        self,
        config: LlamaConfig,
        qkv_proj,
        o_proj,
        rotary_emb,
        name,
        save_data=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb
        self.name = name

        self.save_data = save_data

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        save = False
        if 'model.layers.31' in self.name or 'model.layers.0' in self.name:
            save = True & (self.save_data is not None)
        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = torch.split(qkv_states, self.hidden_size, dim=2)
        if save:
            save_data = self.save_data
            name = self.name
            save_data.add_data(f"{name}_input", hidden_states, "bsz, seqlen, dim")

            q, k, v = torch.split(self.qkv_proj.qweight, self.hidden_size, dim=1)
            save_data.add_data(f"{name}_qweight", q.contiguous(), "dim / 8, dim")
            save_data.add_data(f"{name}_kweight", k.contiguous(), "dim / 8, dim")
            save_data.add_data(f"{name}_vweight", v.contiguous(), "dim / 8, dim")

            q, k, v = torch.split(self.qkv_proj.qzeros, self.hidden_size // 8, dim=1)
            save_data.add_data(f"{name}_qzeros", q.contiguous(), "dim / group_size, dim / 8")
            save_data.add_data(f"{name}_kzeros", k.contiguous(), "dim / group_size, dim / 8")
            save_data.add_data(f"{name}_vzeros", v.contiguous(), "dim / group_size, dim / 8")

            q, k, v = torch.split(self.qkv_proj.scales, self.hidden_size, dim=1)
            save_data.add_data(f"{name}_qscales", q.contiguous(), "dim / group_size, dim")
            save_data.add_data(f"{name}_kscales", k.contiguous(), "dim / group_size, dim")
            save_data.add_data(f"{name}_vscales", v.contiguous(), "dim / group_size, dim")

            save_data.add_data(f"{name}_q_before_rope", q.contiguous(), "bsz, seqlen, dim")
            save_data.add_data(f"{name}_k_before_rope", k.contiguous(), "bsz, seqlen, dim")

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if save:
            save_data.add_data(f"{name}_Q", query_states.contiguous(), "bsz, n_heads, seqlen, head_dim")
            save_data.add_data(f"{name}_K", key_states.contiguous(), "bsz, n_heads, seqlen, head_dim")
            save_data.add_data(f"{name}_V", value_states.contiguous(), "bsz, n_heads, seqlen, head_dim")

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        if use_cache:
            # Since qkv_proj is fused, query_states etc will hold a reference to the original qkv_states tensor
            # which can cause excessive memory usage by the cache. `contiguous` is a convenient way to workaround this.
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        if save:
            save_data.add_data(f"{name}_QK^T", attn_weights, "bsz, n_heads, seq_len, seq_len")
        scale = math.sqrt(self.head_dim)
        if save:
            save_data.add_data(f"{name}_sqrt(dim)", torch.tensor([scale]), "1")
        attn_weights = attn_weights / scale
        if save:
            save_data.add_data(f"{name}_QK^T|sqrt(dim)", attn_weights, "bsz, n_heads, seq_len, seq_len")

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            if save:
                save_data.add_data(f"{name}_QK^T|sqrt(dim)_aftermask", attn_weights, "bsz, n_heads, seq_len, seq_len")
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if save:
            save_data.add_data(f"{name}_softmax(QK^T|sqrt(dim)_aftermask)", attn_weights, "bsz, n_heads, seq_len, seq_len")
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if save:
            save_data.add_data(f"{name}_qkv_output", attn_output, "bsz, seqlen, dim")

        attn_output = self.o_proj(attn_output)
        if save:
            save_data.add_data(f"{name}_oweight", self.o_proj.qweight, "dim / 8, dim")
            save_data.add_data(f"{name}_ozeros", self.o_proj.qzeros, "dim / group_size, dim / 8")
            save_data.add_data(f"{name}_oscales", self.o_proj.scales, "dim / group_size, dim")

        if not output_attentions:
            attn_weights = None
        if save:
            save_data.add_data(f"{name}_output", attn_output, "bsz, seqlen, dim")
        return attn_output, attn_weights, past_key_value