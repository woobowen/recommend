"""Observe actual GRID eager-T5 generation without replacing its beam routine.

Only use this context around one ``model.generate`` invocation, in eval mode.
It does not seed, replay, re-encode, or alter the input/parameters.  T5Stack is
asked to expose its already-computed eager attention.  The caller must verify
OFF/ON outputs and RNG/Shuffle events before admitting a real experimental line.
No attention backend is silently changed here.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from typing import Any, Sequence

import numpy as np

from .records import finalize_paths, quality_diagnostics, reconstruct_parents


def _numpy(tensor: Any) -> np.ndarray:
    """Copy a tensor to CPU; BF16 floats retain their values through FP32 storage."""
    value = tensor.detach().cpu()
    if value.is_floating_point():
        value = value.float()
    return value.numpy().copy()


def recover_history_item_ids(
    input_sids: np.ndarray,
    input_mask: np.ndarray,
    codebooks_h_by_item: np.ndarray,
) -> np.ndarray:
    """Invert the fixed unique H-code SID asset for visible history groups only.

    No SID is regenerated and no input is rewritten.  Hidden/padding groups are
    represented by -1 rather than assigning an item from an invisible target.
    The caller supplies the already-verified asset, in its actual [H, items] axis.
    """
    sids = np.asarray(input_sids)
    mask = np.asarray(input_mask)
    codebooks = np.asarray(codebooks_h_by_item)
    if codebooks.ndim != 2 or codebooks.dtype.kind not in "iu" or min(codebooks.shape) < 1:
        raise ValueError("codebooks must be the fixed integer [H, items] SID asset")
    h = codebooks.shape[0]
    if sids.ndim != 2 or sids.dtype.kind not in "iu" or sids.shape != mask.shape or sids.shape[1] % h:
        raise ValueError("history input and SID asset axes do not match")
    groups = sids.reshape(sids.shape[0], -1, h)
    grouped_mask = mask.reshape(groups.shape)
    if not np.isin(grouped_mask, [0, 1]).all() or np.any(grouped_mask != grouped_mask[:, :, -1:]):
        raise ValueError("only complete visible or hidden SID groups are supported")
    lookup = {tuple(int(code) for code in row): item for item, row in enumerate(codebooks.T)}
    if len(lookup) != codebooks.shape[1]:
        raise ValueError("the fixed full SID asset is not unique; item identity would be ambiguous")
    result = np.full(groups.shape[:2], -1, dtype=np.int64)
    for sample, item in np.argwhere(grouped_mask[:, :, -1] != 0):
        key = tuple(int(code) for code in groups[sample, item])
        if key not in lookup:
            raise ValueError("a visible history SID has no item identity in the fixed asset")
        result[sample, item] = lookup[key]
    return result


def build_encoder_position_map(
    input_sids: np.ndarray,
    input_mask: np.ndarray,
    *,
    num_hierarchies: int,
    sep_enabled: bool,
    user_enabled: bool,
    aggregated: bool,
    item_ids: np.ndarray | None = None,
    source_permutations: np.ndarray | None = None,
    num_embeddings_per_hierarchy: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Map every encoder slot, retaining pads and original lookup-code identity.

    ``source_permutations[b,m,destination]`` is the original code position whose
    looked-up vector occupies the destination.  For Shuffle3, supply H-wide
    permutations with the fourth coordinate explicitly unchanged.  This helper
    never creates random permutations and never calls the encoder.
    """
    sids = np.asarray(input_sids)
    mask = np.asarray(input_mask)
    if sids.ndim != 2 or sids.dtype.kind not in "iu" or sids.shape != mask.shape:
        raise ValueError("input SID/mask must be matching [samples, SID_slots] arrays")
    if num_hierarchies < 1 or sids.shape[1] % num_hierarchies:
        raise ValueError("encoder SID slots are not complete item groups")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("encoder masks must be binary")
    batch, slots = sids.shape
    items = slots // num_hierarchies
    codes = sids.reshape(batch, items, num_hierarchies)
    masks = mask.reshape(batch, items, num_hierarchies)
    if np.any(masks != masks[:, :, -1:]):
        raise ValueError("partial-valid SID groups require data-contract diagnosis, not a remapped mask")
    if item_ids is not None:
        item_ids = np.asarray(item_ids)
    if item_ids is not None and np.asarray(item_ids).shape != (batch, items):
        raise ValueError("item_ids must match the actual [samples, history_slots] axis")
    if aggregated and sep_enabled:
        raise ValueError("the preserved item MLP removes SEP; conflicting position map flags")
    permutations = None if source_permutations is None else np.asarray(source_permutations)
    if permutations is not None:
        if aggregated:
            raise ValueError("this experiment does not authorize MLP+Shuffle")
        if permutations.shape != (batch, items, num_hierarchies):
            raise ValueError("permutation identity must cover every SID coordinate")
        if permutations.dtype.kind not in "iu":
            raise ValueError("source_permutations must have integer coordinates")
        expected = np.arange(num_hierarchies)[None, None, :]
        if not np.array_equal(np.sort(permutations, axis=-1), np.broadcast_to(expected, permutations.shape)):
            raise ValueError("source_permutations contains a non-permutation")
    result = []
    for sample in range(batch):
        positions = []
        if user_enabled:
            positions.append({"role": "user_token", "visible": 1})
        for item in range(items):
            common = {"history_slot": item}
            if item_ids is not None:
                common["item_id"] = int(item_ids[sample, item])
            if aggregated:
                positions.append({
                    **common,
                    "role": "item_aggregation",
                    "original_sids": codes[sample, item].tolist(),
                    "original_code_mask": masks[sample, item].tolist(),
                    "visible": int(masks[sample, item, -1]),
                    "aggregation_axis": "H*E",
                })
            else:
                for destination in range(num_hierarchies):
                    visible = int(masks[sample, item, destination])
                    sampled_source = destination if permutations is None else int(permutations[sample, item, destination])
                    # The original sampler draws for padding groups as well,
                    # but torch.where leaves those groups at their input slots.
                    source = sampled_source if visible else destination
                    position = {
                        **common,
                        "role": "sid_lookup_vector",
                        "destination_code_position": destination,
                        "lookup_code_position": source,
                        "raw_sid_value": int(codes[sample, item, source]),
                        "visible": visible,
                        "sampled_source_code_position": sampled_source,
                        "padding_group_kept_at_input_positions": not bool(visible),
                    }
                    if num_embeddings_per_hierarchy is not None:
                        position["lookup_table_index"] = (
                            int(codes[sample, item, source]) + source * num_embeddings_per_hierarchy
                        ) if visible else 0
                    positions.append(position)
                if sep_enabled:
                    positions.append({**common, "role": "sep", "visible": int(masks[sample, item, -1])})
        for index, position in enumerate(positions):
            position["encoder_position"] = index
        result.append(positions)
    return result


class GenerationAttentionObserver:
    """Finite last-layer, all-head capture with exact prefix-validated parents.

    Example::

        with GenerationAttentionObserver(model, sample_keys=keys) as observer:
            generated, scores = model.generate(...)
            positions = build_encoder_position_map(observer.input_sids,
                observer.input_mask, num_hierarchies=4, sep_enabled=True,
                user_enabled=False, aggregated=False)
            bundle = observer.finish(generated, scores,
                metadata=run_identity, position_map=positions)

    Sample keys should be opaque in any public export.  Stable controlled raw
    identities, checkpoint/SID hashes, permutations/call IDs and effective
    configuration identity belong in ``metadata`` supplied by the main runner.
    """

    def __init__(
        self,
        model: Any,
        *,
        sample_keys: Sequence[str | int],
        selected_indices: Sequence[int] | None = None,
        max_samples: int = 64,
        max_capture_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self.model = model
        self.batch_size = len(sample_keys)
        self.selected = np.asarray(list(range(self.batch_size)) if selected_indices is None else selected_indices, dtype=np.int64)
        if not 0 < len(self.selected) <= max_samples <= 64:
            raise ValueError("attention capture must select 1..64 fixed samples")
        if np.any(self.selected < 0) or np.any(self.selected >= self.batch_size) or len(np.unique(self.selected)) != len(self.selected):
            raise ValueError("selected sample indices are out of bounds or duplicated")
        self.sample_keys = [str(sample_keys[index]) for index in self.selected]
        self.beam_size = int(model.top_k_for_generation)
        self.hierarchies = int(model.num_hierarchies)
        if self.hierarchies != 4 or self.beam_size < 1:
            raise ValueError("this Beauty observer is bounded to a four-code autoregressive model")
        self.max_capture_bytes = max_capture_bytes
        self.steps: list[dict[str, Any]] = []
        self.input_sids = None
        self.input_mask = None
        self.encoder_mask = None
        self._pending = None
        self._handles = []
        self._restore = []
        self._encoder_calls = 0
        self._active = False
        self._finished = False
        self._captured_bytes = 0

    def __enter__(self) -> "GenerationAttentionObserver":
        if self._active or self._finished:
            raise RuntimeError("an attention observer cannot be reused across calls")
        if any(module.training for module in self.model.modules()):
            raise ValueError("attention observation requires the entire model to be in eval mode")
        stack = self.model.decoder.decoder
        if type(stack).__name__ != "T5Stack" or type(stack).__module__ != "transformers.models.t5.modeling_t5":
            raise TypeError("only the actual transformers eager T5Stack is supported")
        implementation = getattr(stack.config, "_attn_implementation", None)
        if implementation not in (None, "eager"):
            raise ValueError(f"attention backend {implementation!r} requires explicit separate verification; no automatic switch")
        cross = stack.block[-1].layer[1].EncDecAttention
        if type(cross).__name__ != "T5Attention":
            raise TypeError("unverified T5 attention module class")
        source = inspect.getsource(type(cross).forward)
        if "scaled_dot_product_attention" in source or "flash_attention" in source:
            raise ValueError("the active attention source may select a different observation backend")
        self.backend = {
            "name": implementation or "eager (source checked)",
            "stack_class": f"{type(stack).__module__}.{type(stack).__name__}",
            "attention_forward_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "attention_module_file": inspect.getfile(type(cross)),
        }
        self._active = True
        try:
            self._handles.append(stack.register_forward_pre_hook(self._request_weights, with_kwargs=True))
            self._handles.append(stack.register_forward_hook(self._capture_weights, with_kwargs=True))
            self._wrap_method("encoder_forward_pass", self._encoder_forward)
            self._wrap_method("_beam_search_one_step", self._beam_step)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wrap_method(self, name: str, observer: Any) -> None:
        sentinel = object()
        previous_attribute = self.model.__dict__.get(name, sentinel)
        original = getattr(self.model, name)
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            return observer(original, bound.arguments, args, kwargs)

        setattr(self.model, name, wrapped)

        def restore() -> None:
            if previous_attribute is sentinel:
                delattr(self.model, name)
            else:
                setattr(self.model, name, previous_attribute)
        self._restore.append(restore)

    def _request_weights(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # Binding preserves positional arguments rather than accidentally supplying
        # output_attentions twice.  The stack config and backend stay unchanged.
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        bound.arguments["output_attentions"] = True
        return bound.args, bound.kwargs

    def _capture_weights(self, module: Any, args: Any, kwargs: Any, output: Any) -> None:
        if self._pending is not None:
            raise RuntimeError("a second decoder call occurred without beam selection")
        attentions = getattr(output, "cross_attentions", None)
        if attentions is None or attentions[-1] is None:
            raise RuntimeError("the requested actual cross-attention was not returned")
        last_layer = attentions[-1]
        if last_layer.ndim != 4 or last_layer.shape[0] % self.batch_size:
            raise ValueError("unexpected decoder cross-attention beam layout")
        parent_beams = int(last_layer.shape[0] // self.batch_size)
        expected = 1 if len(self.steps) == 0 else self.beam_size
        if parent_beams != expected:
            raise ValueError("actual preselection parent beam width differs from GRID generation")
        # Copy only fixed samples, the last layer and the actual last query.
        selected_rows = np.concatenate([index * parent_beams + np.arange(parent_beams) for index in self.selected])
        import torch
        rows = torch.as_tensor(selected_rows, device=last_layer.device)
        values = _numpy(last_layer.index_select(0, rows)[:, :, -1, :])
        values = values.reshape(len(self.selected), parent_beams, last_layer.shape[1], last_layer.shape[-1])
        self._captured_bytes += values.nbytes
        if self._captured_bytes > self.max_capture_bytes:
            raise MemoryError("bounded attention capture byte budget exceeded")
        self._pending = {
            "cross_heads": values,
            "attention_dtype": str(last_layer.dtype),
            "query_length_before_last_selection": int(last_layer.shape[-2]),
            "last_query_index": int(last_layer.shape[-2] - 1),
            "parent_beams": parent_beams,
        }

    def _encoder_forward(self, original: Any, bound: dict[str, Any], args: Any, kwargs: Any) -> Any:
        self._encoder_calls += 1
        if self._encoder_calls != 1:
            raise RuntimeError("capture must surround one generate call, without re-encoding or replay")
        self.input_sids = _numpy(bound["input_ids"])[self.selected]
        self.input_mask = _numpy(bound["attention_mask"])[self.selected]
        if self.input_sids.shape[0] != len(self.selected):
            raise ValueError("sample selection does not match encoder input")
        output = original(*args, **kwargs)
        self.encoder_mask = _numpy(output[1])[self.selected]
        return output

    def _beam_step(self, original: Any, bound: dict[str, Any], args: Any, kwargs: Any) -> Any:
        hierarchy = int(bound["hierarchy"])
        if hierarchy != len(self.steps) or self._pending is None:
            raise RuntimeError("beam observation is not aligned to the actual decoder step")
        previous = bound["generated_ids"]
        previous_ids = None if previous is None else _numpy(previous)[self.selected]
        # Exactly one call, preserving original pruning/scoring/topk/cache logic.
        result = original(*args, **kwargs)
        generated = _numpy(result[0])[self.selected]
        scores = _numpy(result[1])[self.selected]
        if generated.shape != (len(self.selected), self.beam_size, hierarchy + 1):
            raise ValueError("unexpected generated SID shape from original beam routine")
        parents = reconstruct_parents(previous_ids, generated)
        local_prefix = np.empty((len(self.selected), 0), dtype=np.int64) if previous_ids is None else previous_ids[:, 0].copy()
        self.steps.append({
            **self._pending,
            "hierarchy": hierarchy,
            "query_identity": "BOS" if hierarchy == 0 else f"last token in selected prefix of length {hierarchy}",
            "local_first_prefix": local_prefix,
            "parent_indices": parents,
            "generated_ids": generated,
            "scores": scores,
        })
        self._pending = None
        return result

    def finish(
        self,
        generated_ids: Any,
        scores: Any,
        *,
        metadata: dict[str, Any],
        position_map: list[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        if not self._active or self._finished or len(self.steps) != self.hierarchies or self._pending is not None:
            raise RuntimeError("capture has not completed one intact four-step generation")
        final_ids = _numpy(generated_ids)[self.selected]
        final_scores = _numpy(scores)[self.selected]
        if not np.array_equal(final_ids, self.steps[-1]["generated_ids"]) or not np.array_equal(final_scores, self.steps[-1]["scores"]):
            raise ValueError("returned generation does not equal the recorded final beam step")
        if self.encoder_mask is None or not np.isin(self.encoder_mask, [0, 1]).all():
            raise ValueError("missing or nonbinary actual encoder mask")
        if len(position_map) != len(self.selected):
            raise ValueError("position maps do not cover the selected sample set")
        for sample, positions in enumerate(position_map):
            if len(positions) != self.encoder_mask.shape[1]:
                raise ValueError("position map length differs from actual encoder length")
            visible = np.asarray([position["visible"] for position in positions])
            if not np.array_equal(visible, self.encoder_mask[sample]):
                raise ValueError("position map masks differ from actual encoder masks")
        paths = finalize_paths(self.steps)
        diagnostics = quality_diagnostics(paths, self.encoder_mask)
        self._finished = True
        return {
            "capture": {
                "source": "actual generation; no prefix/encoder replay",
                "layer": "last decoder layer",
                "heads": "all heads retained; mean used for display",
                "query": "actual final query at each generation step",
                "parent_mapping": "unique exact prefix identity from unchanged original beam routine",
                "layout": "sample-major, then parent beam",
                "batch_size": self.batch_size,
                "beam_size": self.beam_size,
                "selected_indices": self.selected.tolist(),
                "sample_keys": self.sample_keys,
                "encoder_calls": self._encoder_calls,
                "backend": self.backend,
                "captured_cross_head_bytes": self._captured_bytes,
                "normalization": "raw separately normalized cross-attention; no rescaling",
                "local_first_path": "preselection parent rank 0 independently at each step; not one final SID trajectory",
                "final_top1_path": "final child rank 0 backtracked through each preselection parent",
            },
            "metadata": metadata,
            "input_sids": self.input_sids,
            "input_mask": self.input_mask,
            "encoder_mask": self.encoder_mask,
            "position_map": position_map,
            "steps": self.steps,
            "final_generated_ids": final_ids,
            "final_scores": final_scores,
            "paths": paths,
            "quality": diagnostics,
        }

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # Restore even when generation/attention writing raises; no observation
        # flag can leak into the next test or model call.
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for restore in reversed(self._restore):
            restore()
        self._restore.clear()
        self._active = False
