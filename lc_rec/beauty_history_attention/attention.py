"""Instance-local masked attention for LETTER on transformers 4.31.0.

The decoder still builds position IDs and passes its computed mask explicitly
to every layer, including to the checkpointed call. Generation gets the
unmodified binary padding mask; only the inner decoder sees the encoded mask.
"""

from types import MethodType

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def allowed_edges(valid, history, query_length, mode):
    """Visibility in physical sequence coordinates, including any cached keys."""
    length = valid.shape[1]
    key = torch.arange(length, device=valid.device)
    query = key[length - query_length:]
    edges = key[None, :] <= query[:, None]
    if mode == "bidirectional":
        edges = edges | (history[:, -query_length:, None] & history[:, None, :])
    return edges & valid[:, None, :] & valid[:, -query_length:, None]


def reference_attention(q, k, v, edges):
    """Materialized float32 reference on the same rotated Q/K and V."""
    scores = (q.float() @ k.float().transpose(-2, -1)) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~edges, -torch.inf).masked_fill(~edges.any(-1, keepdim=True), 0)
    probs = scores.softmax(dim=-1).masked_fill(~edges, 0)
    return (probs @ v.float()).to(v.dtype)


class HistoryCache(tuple):
    def __new__(cls, layers, owner, prefix):
        result = super().__new__(cls, layers)
        result.owner = owner
        result.prefix = prefix
        return result


def install_attention(model, mode):
    """Install after model construction, without replacing parameter modules."""
    import transformers
    from transformers.models.llama.modeling_llama import (
        LlamaAttention, LlamaForCausalLM, LlamaModel, apply_rotary_pos_emb, repeat_kv,
    )

    if mode not in ("bidirectional", "causal"):
        raise ValueError(f"unknown history attention mode {mode!r}")
    if transformers.__version__ != "4.31.0" or not isinstance(model, LlamaForCausalLM):
        raise TypeError("expected a transformers 4.31.0 LlamaForCausalLM or LETTER")
    if getattr(model, "_beauty_history_attention", None) is not None:
        raise ValueError("history attention already installed")
    if (LlamaAttention.forward.__module__ != LlamaAttention.__module__
            or LlamaModel._prepare_decoder_attention_mask.__module__ != LlamaModel.__module__):
        raise RuntimeError("reset the global LLaMA attention implementation before installing")
    if model.config.pretraining_tp != 1:
        raise ValueError("the fixed LETTER model uses unsliced Q/K/V/O projections")

    original_forward = model.forward
    original_prepare = model.prepare_inputs_for_generation
    original_reorder = model._reorder_cache
    owner = object()
    report = {"mode": mode, "cuda_backend": "EFFICIENT_ATTENTION only",
              "cpu_backend": "MATH only", "layer_calls": 0, "q1_calls": 0,
              "prefill_calls": 0, "cached_calls": 0, "qkv_dtypes": {},
              "layer_count": len(model.model.layers)}

    def prepare_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        valid = attention_mask.to(device=inputs_embeds.device) != 0
        history = attention_mask.to(device=inputs_embeds.device) == 2
        edges = allowed_edges(valid, history, input_shape[-1], mode).unsqueeze(1)
        # Efficient SDPA requires a multiple-of-eight row stride for BF16 bias.
        key_length = edges.shape[-1]
        columns = (key_length + 7) // 8 * 8
        shape = (*edges.shape[:-1], columns)
        bias = inputs_embeds.new_zeros(shape)[..., :key_length]
        bias.masked_fill_(~edges, -torch.inf)
        # Give an ignored query one finite entry for kernel portability. The
        # attention output for those queries is subsequently forced to zero.
        active = edges.any(dim=-1, keepdim=True)
        bias[..., :1].masked_fill_(~active, 0)
        return bias, active

    def attention_forward(self, hidden_states, attention_mask=None,
                          position_ids=None, past_key_value=None,
                          output_attentions=False, use_cache=False):
        if output_attentions:
            raise ValueError("use the reference calculation to inspect attention weights")
        batch, q_length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, q_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, q_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, q_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_length = q_length + (0 if past_key_value is None else past_key_value[0].shape[-2])
        cos, sin = self.rotary_emb(v, seq_len=key_length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)
        layer_cache = (k, v) if use_cache else None
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        bias, active = attention_mask
        backend = SDPBackend.EFFICIENT_ATTENTION if q.is_cuda else SDPBackend.MATH
        # Casting preserves the bias's aligned row stride when dtypes agree.
        if bias.dtype != q.dtype:
            padded_bias = torch.empty((*bias.shape[:-1], (key_length + 7) // 8 * 8),
                                      dtype=q.dtype, device=q.device)[..., :key_length]
            padded_bias.copy_(bias)
            bias = padded_bias
        with sdpa_kernel(backend):
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        context = context.masked_fill(~active, 0)
        report["layer_calls"] += 1
        report["q1_calls"] += int(q_length == 1)
        dtype = "/".join(str(t.dtype) for t in (q, k, v))
        report["qkv_dtypes"][dtype] = report["qkv_dtypes"].get(dtype, 0) + 1
        context = context.transpose(1, 2).reshape(batch, q_length, self.hidden_size)
        return self.o_proj(context), None, layer_cache

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                history_sid_mask=None):
        inputs = input_ids if input_ids is not None else inputs_embeds
        batch, q_length = inputs.shape[:2]
        cached = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        total = cached + q_length
        if attention_mask is None:
            attention_mask = torch.ones((batch, total), device=inputs.device, dtype=torch.bool)
        if attention_mask.shape != (batch, total):
            raise ValueError("attention_mask must cover the whole physical key sequence")
        valid = attention_mask.bool()
        if history_sid_mask is None:
            raise ValueError("history_sid_mask is required in both experimental modes")
        if (history_sid_mask.ndim != 2 or history_sid_mask.dtype is not torch.bool
                or history_sid_mask.shape[0] != batch or history_sid_mask.shape[1] > total
                or history_sid_mask.device != valid.device):
            raise ValueError("history_sid_mask must be a bool prefix on the input device")
        if labels is not None and history_sid_mask.shape[1] != total:
            raise ValueError("training history mask must span the labelled input")
        history = F.pad(history_sid_mask, (0, total - history_sid_mask.shape[1]))
        # Encodes 0=padding, 1=other valid, 2=valid history. Never leaks to the
        # generation helper, position-ID computation, or other callers.
        encoded = valid.to(torch.uint8) + (valid & history).to(torch.uint8)
        if cached:
            if (not isinstance(past_key_values, HistoryCache) or past_key_values.owner is not owner
                    or not torch.equal(encoded[:, :cached], past_key_values.prefix)):
                raise ValueError("KV cache must originate from this attention instance and the same prefix mask")
            if history_sid_mask.shape[1] > cached and history[:, cached:].any():
                raise ValueError("all bidirectional history must be computed together in prefill")
        result = original_forward(
            input_ids=input_ids, attention_mask=encoded, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )
        report["cached_calls" if cached else "prefill_calls"] += 1
        cache_enabled = self.config.use_cache if use_cache is None else use_cache
        if cache_enabled and self.model.gradient_checkpointing and self.training:
            cache_enabled = False
        if cache_enabled:
            dict_return = self.config.use_return_dict if return_dict is None else return_dict
            if dict_return:
                result.past_key_values = HistoryCache(result.past_key_values, owner, encoded)
            else:
                index = 2 if labels is not None else 1
                cache = HistoryCache(result[index], owner, encoded)
                result = result[:index] + (cache,) + result[index + 1:]
        return result

    def prepare_generation(self, input_ids, past_key_values=None, attention_mask=None,
                           inputs_embeds=None, history_sid_mask=None, **kwargs):
        result = original_prepare(input_ids, past_key_values=past_key_values,
                                  attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        result["history_sid_mask"] = history_sid_mask
        return result

    def reorder_cache(self, past_key_values, beam_idx):
        if not isinstance(past_key_values, HistoryCache) or past_key_values.owner is not owner:
            raise ValueError("beam search received a foreign KV cache")
        layers = original_reorder(past_key_values, beam_idx)
        prefix = past_key_values.prefix
        return HistoryCache(layers, owner, prefix.index_select(0, beam_idx.to(prefix.device)))

    model.model._prepare_decoder_attention_mask = MethodType(prepare_mask, model.model)
    for layer in model.model.layers:
        layer.self_attn.forward = MethodType(attention_forward, layer.self_attn)
    model.forward = MethodType(forward, model)
    model.prepare_inputs_for_generation = MethodType(prepare_generation, model)
    model._reorder_cache = MethodType(reorder_cache, model)
    model._beauty_history_attention = report
    return report
