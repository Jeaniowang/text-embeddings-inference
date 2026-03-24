import os
import torch
import torch_npu
import json
import itertools
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from typing import List, Union, Optional
from safetensors import safe_open
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3 import Qwen3Config
from transformers import AutoTokenizer
from opentelemetry import trace
from text_embeddings_server.models import Model
from text_embeddings_server.models.pooling import DefaultPooling
from text_embeddings_server.models.types import FlashBatch, PaddedBatch, Embedding, Score
from text_embeddings_server.utils.flash_attn import attention

tracer = trace.get_tracer(__name__)
from loguru import logger

def load_weight(model_path, weight_map, prefix, name, dtype, device):
    """
    Helper function to load a weight tensor from safetensors.
    """
    prefix = prefix + "." if prefix else ""
    name = prefix + name
        
    if weight_map is None:
        with safe_open(f"{model_path}/model.safetensors", framework="pt") as f:
            return f.get_tensor(name).to(dtype).to(device)
    else:
        target_file = weight_map[name]
        with safe_open(f"{model_path}/{target_file}", framework="pt") as f:
            return f.get_tensor(name).to(dtype).to(device)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
 
def apply_rotary_pos_emb_npu(q, k, cos, sin, unsqueeze_dim=1):
        
    enable_fp32_compute = False
    def _pre_process(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
            origin_shape = x.shape
            if len(origin_shape) == 3:
                # x: [seq_len, num_heads, head_size]
                x = x.unsqueeze(0)

            origin_dtype = x.dtype
            if enable_fp32_compute:
                x = x.float()
                cos = cos.float()
                sin = sin.float()

            return x, cos, sin, origin_shape, origin_dtype
        
    def _post_process(
        output: torch.Tensor,
        origin_shape: torch.Size,
        origin_dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(origin_shape) == 3:
            output = output.squeeze(0)
        if enable_fp32_compute:
            output = output.to(origin_dtype)
        return output
    
    # q_ [1, seq_len, num_heads, head_size]
    # k_ [1, seq_len, num_heads // 2, head_size]
    q_, cos, sin, q_origin_shape, q_origin_dtype = _pre_process(q, cos, sin)
    k_, cos, sin, k_origin_shape, k_origin_dtype = _pre_process(k, cos, sin)
    
    head_dim = q_.shape[-1]

    # cos, sin: [1, seq_len, 1, head_dim]
    cos = cos.reshape(1, -1, 1, head_dim)
    sin = sin.reshape(1, -1, 1, head_dim)
    output_q = torch_npu.npu_rotary_mul(q_, cos, sin)
    output_k = torch_npu.npu_rotary_mul(k_, cos, sin)

    output_q = _post_process(output_q, q_origin_shape, q_origin_dtype)
    output_k = _post_process(output_k, k_origin_shape, k_origin_dtype)

    return output_q, output_k


def compute_default_rope_parameters(
    config: Qwen3Config,
    device: torch.device,
) -> tuple["torch.Tensor", float]:
    base = config.rope_theta
    partial_rotary_factor = (
        config.partial_rotary_factor
        if hasattr(config, "partial_rotary_factor")
        else 1.0
    )
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, attention_factor


class Qwen3RMSNorm:
    def __init__(
        self,
        model_path,
        weight_map,
        prefix,
        name,
        device,
        dtype,
        eps=1e-6,
    ):
        self.weight = load_weight(model_path, weight_map, prefix, name, dtype, device)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if hidden_states.device.type == "hpu":
            from habana_frameworks.torch.hpex.normalization import (
                FusedRMSNorm as FusedRMSNorm,
            )

            hidden_states = FusedRMSNorm.apply(
                hidden_states, self.weight, self.variance_epsilon
            )
            return hidden_states
        elif hidden_states.device.type == "npu":
            return torch_npu.npu_rms_norm(hidden_states,
                                          self.weight, epsilon = self.variance_epsilon)[0]
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return self.weight * hidden_states.to(input_dtype)


class Qwen3Attention:
    def __init__(
        self,
        model_path,
        weight_map,
        prefix,
        device,
        dtype,
        config: Qwen3Config,
        layer_idx: Optional[int] = None,
    ):
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.softmax_scale = self.head_dim**-0.5
        self.q_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.q_proj.weight",
            dtype,
            device,
        )
        self.k_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.k_proj.weight",
            dtype,
            device,
        )
        self.v_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.v_proj.weight",
            dtype,
            device,
        )
        self.qkv_weight = (
            torch.cat([self.q_proj_weight, self.k_proj_weight, self.v_proj_weight]).to(dtype).to(device)
        )
        self.o_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.o_proj.weight",
            dtype,
            device,
        )
        self.q_norm = Qwen3RMSNorm(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.q_norm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Qwen3RMSNorm(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.k_norm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )

    def forward(
        self, hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask=None
    ):
        input_shape = hidden_states.shape[:-1]

        q = self.q_norm.forward(
            F.linear(hidden_states, self.q_proj_weight).view(*input_shape, self.num_heads, self.head_dim)
        )
        k = self.k_norm.forward(
            F.linear(hidden_states, self.k_proj_weight).view(*input_shape, self.num_key_value_heads, self.head_dim)
        )
        v = F.linear(hidden_states, self.v_proj_weight).view(*input_shape, self.num_key_value_heads, self.head_dim)
        
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_npu(q, k, cos, sin, unsqueeze_dim=1)

        if self.num_key_value_groups > 1:
        # 先扩展一个维度，然后 reshape 实现元素级重复
            k = k.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1).reshape(k.shape[0], -1, k.shape[2])
            v = v.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1).reshape(v.shape[0], -1, v.shape[2])
        attn_output = attention(
            q,
            k,
            v,
            self.num_heads,
            None,
            cu_seqlens,
            max_s,
            self.softmax_scale,
            is_causal=True,
            attn_mask=attn_mask,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = F.linear(attn_output, self.o_proj_weight, bias=None)

        return attn_output


class Qwen3MLP:
    def __init__(
        self,
        model_path,
        weight_map,
        prefix,
        device,
        dtype,
        config: Qwen3Config,
        layer_idx: Optional[int] = None,
    ):
        self.gate_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.mlp.gate_proj.weight",
            dtype,
            device,
        )
        self.up_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.mlp.up_proj.weight",
            dtype,
            device,
        )
        self.down_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.mlp.down_proj.weight",
            dtype,
            device,
        )
        self.act_fn = ACT2FN[config.hidden_act]


    def forward(self, hidden_state):
        gated_hidden_states = F.linear(hidden_state, self.gate_proj_weight)
        uped_hidden_states = F.linear(hidden_state, self.up_proj_weight)
        return F.linear(
            self.act_fn(gated_hidden_states) * uped_hidden_states,
            self.down_proj_weight,
        )

class Qwen3DecoderLayer:
    def __init__(
        self,
        model_path,
        weight_map,
        prefix,
        device,
        dtype,
        config: Qwen3Config,
        layer_idx: Optional[int] = None,
    ):
        self.config = config
        self.attention = Qwen3Attention(
            model_path, weight_map, prefix, device, dtype, config, layer_idx
        )
        self.mlp = Qwen3MLP(model_path, weight_map, prefix, device, dtype, config, layer_idx)
        self.input_layernorm = Qwen3RMSNorm(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.input_layernorm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.post_attention_layernorm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )

    def forward(
        self, hidden_states, residual, position_embeddings, cu_seqlens, max_s, attn_mask=None
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm.forward(hidden_states)
        else:
            hidden_states, _, residual = torch_npu.npu_add_rms_norm(hidden_states, residual, self.input_layernorm.weight, epsilon=self.config.rms_norm_eps)
        # Self Attention
        hidden_states = self.attention.forward(
            hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask
        )
  
        hidden_states, _, residual = torch_npu.npu_add_rms_norm(hidden_states, residual, self.post_attention_layernorm.weight, epsilon=self.config.rms_norm_eps)
        hidden_states = self.mlp.forward(hidden_states)
        return hidden_states, residual


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        inv_freq, self.attention_scaling = compute_default_rope_parameters(
            config, device
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        # inv_freq     [1, head_dim // 2]
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FlashQwen3Model:
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, model_path, weight_map, prefix, device, dtype, config: Qwen3Config):
        self.word_embeddings_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            "embed_tokens.weight",
            dtype,
            device,
        )
        self.config = config
        self.layers = [
            Qwen3DecoderLayer(
                model_path,
                weight_map,
                prefix,
                device,
                dtype,
                config,
                layer_idx,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.rotary_emb = Qwen3RotaryEmbedding(config=config, device=device)
        self.norm = Qwen3RMSNorm(
            model_path,
            weight_map,
            prefix,
            f"norm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        mask=None,
        attn_mask=None,
    ):
        inputs_embeds = nn.functional.embedding(input_ids, self.word_embeddings_weight)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer.forward(
                hidden_states, residual, position_embeddings, cu_seqlens, max_s, attn_mask
            )
        hidden_states, _, _ = torch_npu.npu_add_rms_norm(hidden_states, residual, self.norm.weight, epsilon=self.config.rms_norm_eps)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class ForSequenceClassification(nn.Module):
    def __init__(self, model_path, weight_map, prefix, device, dtype, config: Qwen3Config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = FlashQwen3Model(model_path, weight_map, prefix,device, dtype, config)
 
        self.word_embeddings_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            "embed_tokens.weight",
            dtype,
            device,
        )
        
        if hasattr(self.config, "max_seq_length"):
            self.max_input_length = self.config.max_seq_length
        else:
            self.max_input_length = self.config.max_position_embeddings

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
                 
    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        mask=None,
        attn_mask=None,
    ):
        input_ids_list = input_ids.tolist()
        cu_seqlens_list = cu_seqlens.tolist()
        input_ids_list_new = []
        seqlens_list_new = [0]
        position_ids_list = []

        prev_end = 0
        for end in cu_seqlens_list[1:]:
            sequence = input_ids_list[prev_end:end]
            new_sequence = self.prefix_tokens + sequence + self.suffix_tokens
            input_ids_list_new.extend(new_sequence)

            seqlens_list_new.append(len(new_sequence))
            position_ids_list.extend(range(len(new_sequence)))
            prev_end = end
        cu_seqlens_list_new = list(itertools.accumulate(seqlens_list_new))
        
        input_ids_list_new_tensor = torch.tensor(input_ids_list_new, device=input_ids.device)
        cu_seqlens_list_new_tensor = torch.tensor(cu_seqlens_list_new, device=cu_seqlens.device)
        position_ids_list_tensor = torch.tensor(position_ids_list, device=input_ids.device)
        
        output = self.model.forward(input_ids_list_new_tensor,
                position_ids_list_tensor,
                cu_seqlens_list_new_tensor,
                max_s,
                mask,
                attn_mask)
        
        last_token_indices = cu_seqlens_list_new_tensor[1:] - 1
        hidden_states = output.last_hidden_state
        
        logits = F.linear(hidden_states[last_token_indices], self.word_embeddings_weight).cpu() 
        true_logits = logits[:, self.token_true_id]
        false_logits = logits[:, self.token_false_id]
        logit_diff = true_logits - false_logits
        
        return logit_diff
    
class FlashQwen3(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "lasttoken",
        trust_remote: bool = False,
    ):
        config = Qwen3Config.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if hasattr(config, "max_seq_length"):
            self.max_input_length = config.max_seq_length
        else:
            self.max_input_length = config.max_position_embeddings

        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index_data = json.load(f)
            weight_map = index_data["weight_map"]
        else:
            weight_map = None
            
        if os.getenv("IS_RERANK", None):
            model = ForSequenceClassification(model_path, weight_map, "model", device, dtype, config, tokenizer)
        else:    
            model = FlashQwen3Model(model_path, weight_map, "", device, dtype, config)
                
        self.hidden_size = config.hidden_size
        self.pooling = DefaultPooling(self.hidden_size, pooling_mode=pool)
        self.device = device
        self.dtype = dtype

        super(FlashQwen3, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Union[FlashBatch, PaddedBatch]:
        # for hpu devices, we use PaddedBatch as we do not have real varlen fwd yet
        return FlashBatch if self.device.type != "hpu" else PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: Union[FlashBatch, PaddedBatch]) -> List[Embedding]:
        if isinstance(batch, PaddedBatch):
            input_lens = batch.attention_mask.cumsum(-1)[:, -1].to(torch.int32)
            max_input_lens = 0
            cu_seqlens = torch.cat(
                (input_lens.new_tensor([0]), input_lens.cumsum(-1).int())
            )
            mask = batch.attention_mask.bool()
            bsz, tgt_len = mask.size()
            min_val = torch.finfo(self.dtype).min
            attn_mask = torch.full(
                [bsz, 1, tgt_len, tgt_len],
                fill_value=min_val,
                device=self.device,
                dtype=self.dtype,
            )
            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, tgt_len)
            attn_mask = attn_mask.masked_fill(expanded_mask, 0.0)
        elif isinstance(batch, FlashBatch):
            cu_seqlens = batch.cu_seqlens
            mask = None
            attn_mask = None
            max_input_lens = batch.max_s

        output = self.model.forward(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_input_lens,
            mask=mask,
            attn_mask=attn_mask,
        )
        
        last_token_indices = cu_seqlens[1:] - 1
        hidden_states = torch.index_select(output.last_hidden_state, 0, last_token_indices.to(output.last_hidden_state.device))
        cpu_results = hidden_states.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]
        
    @tracer.start_as_current_span("predict")
    def predict(self, batch: Union[FlashBatch, PaddedBatch]) -> List[Score]:
        if isinstance(batch, PaddedBatch):
            input_lens = batch.attention_mask.cumsum(-1)[:, -1].to(torch.int32)
            max_input_lens = 0  # This value will not be used
            cu_seqlens = torch.cat(
                (input_lens.new_tensor([0]), input_lens.cumsum(-1).int())
            )
            mask = batch.attention_mask.bool()
            bsz, tgt_len = mask.size()
            min_val = torch.finfo(self.dtype).min
            attn_mask = torch.full(
                [bsz, 1, tgt_len, tgt_len],
                fill_value=min_val,
                device=self.device,
                dtype=self.dtype,
            )
            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, tgt_len)
            attn_mask = attn_mask.masked_fill(expanded_mask, 0.0)
        elif isinstance(batch, FlashBatch):
            cu_seqlens = batch.cu_seqlens
            mask = None
            attn_mask = None
            max_input_lens = batch.max_s

        logits = self.model.forward(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_input_lens,
            mask=mask,
            attn_mask=attn_mask,
        )

        return [Score(values=[p.item()]) for p in logits]