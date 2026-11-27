"""Native Transformers eager inference with the experiment's additive mask."""

import hashlib
import inspect
from pathlib import Path
from types import MethodType


# Actual source used by the fixed historical LETTER inference environment.
LLAMA_SOURCE_SHA256 = "2abf84a5dd3644330f93c88e036831578e407de7a9812b5454ef1bb8a18c3422"


def install_eager_inference(model, mode):
    """Reuse the history/cache routing while retaining native layer attention."""
    from .attention import install_attention

    if model.training:
        raise ValueError("native eager adaptation is an inference-only path")
    return install_attention(model, mode, inference_backend="native_eager")


def bind_native_eager(decoder, mode, report):
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel

    path = Path(inspect.getsourcefile(LlamaAttention))
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if source_hash != LLAMA_SOURCE_SHA256:
        raise RuntimeError(f"unexpected native LLaMA source: {source_hash}")
    native_prepare = decoder._prepare_decoder_attention_mask
    if native_prepare.__func__ is not LlamaModel._prepare_decoder_attention_mask:
        raise RuntimeError("expected the unmodified native decoder mask function")

    for layer in decoder.layers:
        attention = layer.self_attn
        forward = inspect.unwrap(attention.forward)
        if getattr(forward, "__func__", forward) is not LlamaAttention.forward:
            raise RuntimeError("expected native LlamaAttention, including through Accelerate hooks")

    report.update(backend="native_eager", cuda_backend="Transformers native eager",
                  cpu_backend="Transformers native eager", native_source_sha256=source_hash,
                  native_source=str(path), mask_calls=0, bidirectional_mask_calls=0,
                  cache_reorders=0, native_mask_dtypes={})
    report.pop("qkv_dtypes")

    def prepare_mask(self, encoded, input_shape, inputs_embeds, past_key_values_length):
        if self.training:
            raise RuntimeError("native eager history mask is inference only")
        valid = encoded != 0
        bias = native_prepare(valid, input_shape, inputs_embeds, past_key_values_length)
        query_length = input_shape[-1]
        if mode == "bidirectional" and query_length > 1:
            # Native bias is zero for each already-visible valid history key.
            # Only the intersection of valid history queries and keys changes.
            # Encoded padding is zero, never two. Native behavior for invalid
            # query rows, ordinary padding keys, and target tokens is retained.
            history = encoded == 2
            history_edges = history[:, -query_length:, None] & history[:, None, :]
            bias.masked_fill_(history_edges.unsqueeze(1), 0.0)
            report["bidirectional_mask_calls"] += 1
        report["mask_calls"] += 1
        dtype = str(bias.dtype)
        report["native_mask_dtypes"][dtype] = report["native_mask_dtypes"].get(dtype, 0) + 1
        return bias

    def count_layer(module, args, kwargs):
        states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        report["layer_calls"] += 1
        report["q1_calls"] += int(states.shape[1] == 1)

    decoder._prepare_decoder_attention_mask = MethodType(prepare_mask, decoder)
    for layer in decoder.layers:
        layer.self_attn.register_forward_pre_hook(count_layer, with_kwargs=True)
