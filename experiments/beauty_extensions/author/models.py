"""Minimal runtime bridges; all research parameters come from original classes.

No checkpoint migration or training defaults are supplied here.  A caller must
instantiate a confirmed model configuration and use strict state-dict loading.
The preserved source files remain unmodified and are checked before import.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable

from .bootstrap import assert_author_namespace, bind_author_source

SOURCE_IDENTITY = bind_author_source()

import torch  # noqa: E402

from src.models.components.interfaces import OneKeyPerPredictionOutput  # noqa: E402
from src.models.modules.semantic_id.tiger_generation_model_mlp import (  # noqa: E402
    SemanticIDEncoderDecoder as OriginalMLPModel,
)
from src.models.modules.semantic_id.tiger_generation_model_shuffle import (  # noqa: E402
    SemanticIDEncoderDecoder as OriginalShuffleModel,
)


def validate_item_groups(attention_mask: torch.Tensor, hierarchies: int) -> None:
    """Diagnose unsupported partial groups instead of altering mask semantics."""
    if attention_mask.ndim != 2 or attention_mask.shape[1] % hierarchies:
        raise ValueError("SID history must be a [B, M*H] tensor with complete groups")
    if torch.any((attention_mask != 0) & (attention_mask != 1)):
        raise ValueError("SID attention mask must contain only zero and one")
    grouped = attention_mask.reshape(attention_mask.shape[0], -1, hierarchies)
    partial = (grouped != grouped[:, :, -1:]).any(dim=-1)
    if torch.any(partial):
        raise ValueError(
            f"Partial valid SID groups are unsupported: {int(partial.sum().item())}. "
            "The author mask convention is preserved; repair the data contract first."
        )


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(part) for part in value)
    return value


class _RuntimeBridge:
    """Bridge Lightning contracts and observe outputs without changing the model."""

    def __init__(self, *args, save_attentions_on_test=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_testing = False
        # The original MLP constructor has no such keyword, whereas the
        # original Shuffle class does. Consume this runtime-only bridge option
        # here; never forward an unknown keyword to TransformerBaseModule.
        self.save_attentions_on_test = bool(save_attentions_on_test)
        self.evaluation_sink: Callable | None = None
        self.attention_sink: Callable | None = None
        self.permutation_trace_limit = 4
        self.encoder_trace_limit = 64
        self.encoder_call_count = 0
        self.encoder_role_counts: dict[str, int] = {}
        self.shuffle_call_count = 0
        self.encoder_events: list[dict] = []
        self.permutation_events: list[dict] = []
        self._call_role = "direct"
        self._replay_queue = None
        self._current_encoder_call = None

    def configure_observation(
        self,
        *,
        evaluation_sink: Callable | None = None,
        attention_sink: Callable | None = None,
        permutation_trace_limit: int = 4,
        encoder_trace_limit: int = 64,
    ) -> None:
        if permutation_trace_limit < 0 or encoder_trace_limit < 0:
            raise ValueError("observation trace limits must be nonnegative")
        self.evaluation_sink = evaluation_sink
        self.attention_sink = attention_sink
        self.permutation_trace_limit = permutation_trace_limit
        self.encoder_trace_limit = encoder_trace_limit

    @contextmanager
    def _role(self, role):
        previous = self._call_role
        self._call_role = role
        try:
            yield
        finally:
            self._call_role = previous

    def model_step(self, model_input, label_data=None):
        with self._role("loss" if label_data is not None else "prediction"):
            return super().model_step(model_input=model_input, label_data=label_data)

    def generate(self, *args, **kwargs):
        with self._role("generate"):
            return super().generate(*args, **kwargs)

    def encoder_forward_pass(self, attention_mask, input_ids, user_id=None,
                             output_attentions=False):
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same [B,M*H] shape")
        validate_item_groups(attention_mask, self.num_hierarchies)
        self.encoder_call_count += 1
        self.encoder_role_counts[self._call_role] = self.encoder_role_counts.get(self._call_role, 0) + 1
        call_id = self.encoder_call_count
        self._current_encoder_call = call_id
        shuffled = bool(self.is_testing and getattr(self, "shuffle_embedding_on_test", False))
        self.shuffle_call_count += int(shuffled)
        if len(self.encoder_events) < self.encoder_trace_limit:
            self.encoder_events.append({
                "call_id": call_id,
                "role": self._call_role,
                "input_shape": list(input_ids.shape),
                "is_testing": bool(self.is_testing),
                "shuffle_enabled": shuffled,
                "hierarchies_to_shuffle": getattr(self, "hierarchies_to_shuffle", None) if shuffled else None,
            })
        try:
            return super().encoder_forward_pass(
                attention_mask=attention_mask, input_ids=input_ids, user_id=user_id,
                output_attentions=output_attentions,
            )
        finally:
            self._current_encoder_call = None

    def eval_step(self, batch, loss_to_aggregate, batch_idx=0, save_attentions=False):
        """Preserve the author's loss -> generate -> evaluator order and calls.

        A normal runner leaves save_attentions=False and may install an external
        generation observer.  The explicit legacy flag is also implemented; it
        needs a sink so observations never silently disappear or overwrite old
        cwd-relative attention files.
        """
        model_input, label_data = batch
        if save_attentions and self.attention_sink is None:
            raise RuntimeError("save_attentions=True requires an explicit attention_sink")
        before = self.encoder_call_count
        _, loss = self.model_step(model_input=model_input, label_data=label_data)
        generated = self.generate(
            attention_mask=model_input.mask,
            output_attentions=bool(save_attentions),
            **{self.feature_to_model_input_map.get(k, k): v
               for k, v in model_input.transformed_sequences.items()},
        )
        generated_ids, marginal_probs = generated[:2]
        labels = list(label_data.labels.values())[0].to(marginal_probs.device)
        self.evaluator(marginal_probs=marginal_probs, generated_ids=generated_ids, labels=labels)
        loss_to_aggregate(loss)
        if self.evaluation_sink is not None:
            self.evaluation_sink({
                "batch_idx": int(batch_idx),
                "user_keys": _cpu(model_input.user_id_list),
                "input_sequences": _cpu(model_input.transformed_sequences),
                "input_mask": _cpu(model_input.mask),
                "labels": _cpu(labels),
                "generated_ids": _cpu(generated_ids),
                "scores": _cpu(marginal_probs),
                "loss": float(loss.detach().cpu()),
                "encoder_call_ids": list(range(before + 1, self.encoder_call_count + 1)),
            })
        if save_attentions:
            self.attention_sink({
                "batch_idx": int(batch_idx),
                "user_keys": _cpu(model_input.user_id_list),
                "generated_ids": _cpu(generated_ids), "scores": _cpu(marginal_probs),
                "encoder_attentions": _cpu(generated[2]),
                "cross_attentions": _cpu(generated[3]),
                "decoder_self_attentions": _cpu(generated[4]),
                "path_identity": "raw_per_step_parent_beams; no final-path assertion",
            })
        return generated_ids, marginal_probs

    def test_step(self, batch, batch_idx):
        return self.eval_step(
            batch, self.test_loss, batch_idx=batch_idx,
            save_attentions=bool(getattr(self, "save_attentions_on_test", False)),
        )

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, self.val_loss, batch_idx=batch_idx, save_attentions=False)

    def predict_step(self, batch, batch_idx=0):
        """Avoid the original implicit all-attention cwd writes in predict."""
        generated_ids, _ = self.generate(
            attention_mask=batch.mask,
            **{self.feature_to_model_input_map.get(k, k): v
               for k, v in batch.transformed_sequences.items()},
        )
        keys = [key.item() if isinstance(key, torch.Tensor) else key
                for key in batch.user_id_list]
        return OneKeyPerPredictionOutput(
            keys=keys, predictions=generated_ids, key_name=self.prediction_key_name,
            prediction_name=self.prediction_value_name,
        )

    def on_test_start(self):
        super().on_test_start()
        self.is_testing = True

    def on_test_end(self):
        try:
            return super().on_test_end()
        finally:
            self.is_testing = False
            self._replay_queue = None

    def on_validation_start(self):
        self.is_testing = False
        return super().on_validation_start()

    def on_predict_start(self):
        self.is_testing = False
        return super().on_predict_start()

    def on_train_start(self):
        self.is_testing = False
        return super().on_train_start()

    def clear_runtime_state(self):
        """Called by a runner's finally block, including Trainer exceptions."""
        self.is_testing = False
        self._replay_queue = None
        self._call_role = "direct"
        self._current_encoder_call = None

    def on_exception(self, exception):
        self.clear_runtime_state()
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            trainer.should_stop = True
            for logger in trainer.loggers:
                logger.finalize(status="failure")


class AuthorShuffleModel(_RuntimeBridge, OriginalShuffleModel):
    """Route A: original dual SID tables and original per-call random sampler."""

    @contextmanager
    def replay_permutations(self, permutations: Iterable[torch.Tensor]):
        """Controlled engineering replay only; no RNG draws in this context."""
        if self._replay_queue is not None:
            raise RuntimeError("Nested permutation replay is not supported")
        if not self.is_testing or not self.shuffle_embedding_on_test:
            raise RuntimeError("Replay requires the test-stage shuffle path to be active")
        self._replay_queue = list(permutations)
        try:
            yield
            if self._replay_queue:
                raise RuntimeError(f"Unused replay permutations: {len(self._replay_queue)}")
        finally:
            self._replay_queue = None

    def _shuffle_hierarchical_embeddings(
        self, id_embeddings, attention_mask, num_hierarchies, hierarchies_to_shuffle=3,
    ):
        validate_item_groups(attention_mask, num_hierarchies)
        if hierarchies_to_shuffle not in (3, 4) or hierarchies_to_shuffle > num_hierarchies:
            raise ValueError("This experiment supports Shuffle3/Shuffle4 within H only")
        batch, tokens, hidden = id_embeddings.shape
        if tuple(attention_mask.shape) != (batch, tokens):
            raise ValueError("Embedding/mask shapes do not match")
        items = tokens // num_hierarchies
        n = hierarchies_to_shuffle
        capture = len(self.permutation_events) < self.permutation_trace_limit
        permutation = None
        if self._replay_queue is not None:
            if not self._replay_queue:
                raise RuntimeError("Permutation replay exhausted before encoder calls finished")
            permutation = self._replay_queue.pop(0).to(device=id_embeddings.device)
            if permutation.dtype != torch.int64 or tuple(permutation.shape) != (batch, items, n):
                raise ValueError("Replay permutation must be int64 [B,M,N]")
            expected = torch.arange(n, device=id_embeddings.device).expand(batch, items, n)
            if not torch.equal(permutation.sort(dim=-1).values, expected):
                raise ValueError("Replay rows must be permutations of range(N)")
            grouped = id_embeddings.clone().reshape(batch, items, num_hierarchies, hidden)
            active = attention_mask.reshape(batch, items, num_hierarchies).sum(dim=-1) > 0
            shuffled = grouped[:, :, :n].gather(
                2, permutation.unsqueeze(-1).expand(batch, items, n, hidden),
            )
            grouped[:, :, :n] = torch.where(active[:, :, None, None], shuffled, grouped[:, :, :n])
            result = grouped.reshape(batch, tokens, hidden)
            protocol = "engineering_replay"
        else:
            # Capture the default generator state immediately before the exact
            # original helper. Reconstruct trace via an independent generator;
            # this does not consume or reset the global random stream.
            generator = None
            if capture:
                generator = torch.Generator(device=id_embeddings.device)
                state = (torch.cuda.get_rng_state(id_embeddings.device)
                         if id_embeddings.is_cuda else torch.get_rng_state())
                generator.set_state(state)
            result = super()._shuffle_hierarchical_embeddings(
                id_embeddings=id_embeddings.clone(), attention_mask=attention_mask,
                num_hierarchies=num_hierarchies, hierarchies_to_shuffle=n,
            )
            if capture:
                permutation = torch.rand(
                    batch, items, n, device=id_embeddings.device, generator=generator,
                ).argsort(dim=2)
            protocol = "original_rand_argsort_per_encoder_call"
        if capture:
            self.permutation_events.append({
                "encoder_call_id": self._current_encoder_call,
                "role": self._call_role,
                "shape": [batch, items, n], "protocol": protocol,
                "permutation": permutation.detach().cpu().tolist(),
                "item_mask": attention_mask.reshape(batch, items, num_hierarchies)[:, :, -1]
                    .detach().cpu().tolist(),
                "trace_rng": "independent generator; global stream unchanged by tracing",
            })
        return result


class AuthorMLPModel(_RuntimeBridge, OriginalMLPModel):
    """Original ordered per-item MLP, last-code mask, no SEP, shared SID table."""


assert_author_namespace()
