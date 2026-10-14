"""Route B: observe the shared SID lookup only inside encoder_forward_pass.

No parameter, decoder, beam, evaluator, or embedding sharing is replaced. The
intervention calls the frozen author's function body, extracted with its file
identity recorded by the run. OFF returns the original lookup tensor untouched.
"""
import ast
import contextlib
import hashlib
from pathlib import Path

import torch
from src.models.modules.semantic_id.tiger_generation_model import SemanticIDEncoderDecoder


NATIVE_FILE = Path(__file__).resolve().parents[2] / "references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py"


def native_shuffle():
    tree = ast.parse(NATIVE_FILE.read_bytes(), filename=str(NATIVE_FILE))
    methods = [n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
               if isinstance(n, ast.FunctionDef) and n.name == "_shuffle_hierarchical_embeddings"]
    if len(methods) != 1:
        raise RuntimeError("Frozen native shuffle method is not unique")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(NATIVE_FILE), "exec"), namespace)
    return namespace[methods[0].name]


NATIVE_SHUFFLE = native_shuffle()


class BaselineShuffleModel(SemanticIDEncoderDecoder):
    def __init__(self, shuffle_n=0, **kwargs):
        super().__init__(**kwargs)
        if shuffle_n not in (0, 3, 4):
            raise ValueError("shuffle_n must be OFF, 3, or 4")
        self.shuffle_n = shuffle_n
        self.is_testing = False
        self.encoder_calls = 0
        self.encoder_role_counts = {}
        self.shuffle_role_counts = {}
        self.shuffle_summary = {"sampled_items_including_padding": 0, "identity_permutations": 0, "calls": 0}
        self._role = "direct"
        self.shuffle_events = []
        self.trace_limit = 6
        self._encoder_mask = None
        self._replay = None
        self.capture_full_permutation = False
        self.last_full_permutation = None
        self._lookup_hook = self.item_sid_embedding_table_encoder.register_forward_hook(self._lookup)

    def _lookup(self, module, args, output):
        if self._encoder_mask is None:
            return output
        self.encoder_calls += 1
        self.encoder_role_counts[self._role] = self.encoder_role_counts.get(self._role, 0) + 1
        if not self.is_testing or self.shuffle_n == 0:
            return output
        b, length, e = output.shape
        h = self.num_hierarchies
        if h != 4 or length % h:
            raise ValueError("Expected complete four-code item groups")
        mask = self._encoder_mask.reshape(b, -1, h)
        if not torch.all((mask == mask[..., :1])):
            raise ValueError("Partial-valid item groups are unsupported; mask semantics unchanged")
        n = self.shuffle_n
        before = output.detach().clone() if len(self.shuffle_events) < self.trace_limit else None
        if self._replay is not None:
            perm = self._replay.to(output.device)
            if tuple(perm.shape) != (b, length // h, n):
                raise ValueError("Replay must be [B,M,N]")
            expected = torch.arange(n, device=output.device).expand_as(perm)
            if not torch.equal(perm.sort(-1).values, expected):
                raise ValueError("Invalid replay permutation")
            grouped = output.reshape(b, -1, h, e).clone()
            changed = grouped[..., :n, :].gather(2, perm[..., None].expand(-1, -1, -1, e))
            grouped[..., :n, :] = torch.where(mask[..., :1, None].bool(), changed, grouped[..., :n, :])
            result = grouped.reshape_as(output)
        else:
            # Reconstruct trace with an isolated generator, never advance global RNG.
            generator = torch.Generator(device=output.device)
            state = torch.cuda.get_rng_state(output.device) if output.is_cuda else torch.get_rng_state()
            generator.set_state(state)
            perm = torch.rand(b, length // h, n, device=output.device, generator=generator).argsort(2)
            result = NATIVE_SHUFFLE(self, output, self._encoder_mask, h, n)
        if before is not None:
            before_groups, after_groups = before.reshape(b, -1, h, e), result.reshape(b, -1, h, e)
            assert torch.equal(before_groups.sort(2).values, after_groups.sort(2).values)
            assert torch.equal(before_groups[mask.sum(-1) == 0], after_groups[mask.sum(-1) == 0])
            if n == 3:
                assert torch.equal(before_groups[:,:,3], after_groups[:,:,3])
            self.shuffle_events.append({"encoder_call": self.encoder_calls, "shape": [b, length // h, n],
                "role": self._role, "n": n, "multiset_exact": True, "padding_unchanged": True,
                "replay": self._replay is not None, "permutations": perm[:2, :6].cpu().tolist()})
        if self.capture_full_permutation:
            self.last_full_permutation = perm.detach().cpu().clone()
        self.shuffle_role_counts[self._role] = self.shuffle_role_counts.get(self._role, 0) + 1
        self.shuffle_summary["calls"] += 1
        self.shuffle_summary["sampled_items_including_padding"] += b * (length // h)
        self.shuffle_summary["identity_permutations"] += int((perm == torch.arange(n, device=perm.device)).all(-1).sum())
        return result

    def model_step(self, model_input, label_data=None):
        old = self._role
        self._role = "loss" if label_data is not None else "prediction"
        try:
            return super().model_step(model_input=model_input, label_data=label_data)
        finally:
            self._role = old

    def generate(self, *args, **kwargs):
        old = self._role
        self._role = "generate"
        try:
            return super().generate(*args, **kwargs)
        finally:
            self._role = old

    def encoder_forward_pass(self, attention_mask, input_ids, user_id=None):
        if self._encoder_mask is not None:
            raise RuntimeError("Nested encoder scope")
        self._encoder_mask = attention_mask
        try:
            return super().encoder_forward_pass(attention_mask, input_ids, user_id)
        finally:
            self._encoder_mask = None

    @contextlib.contextmanager
    def replay_permutations(self, permutations):
        if self._replay is not None:
            raise RuntimeError("Nested permutation replay")
        self._replay = permutations
        try:
            yield
        finally:
            self._replay = None

    def on_test_start(self):
        super().on_test_start()
        self.is_testing = True

    def on_test_end(self):
        try:
            super().on_test_end()
        finally:
            self.is_testing = False

    def on_train_start(self):
        self.is_testing = False
        return super().on_train_start()

    def on_validation_start(self):
        self.is_testing = False
        return super().on_validation_start()

    def on_exception(self, exception):
        self.is_testing = False
        self._encoder_mask = None

    def native_identity(self):
        return {"path": str(NATIVE_FILE), "sha256": hashlib.sha256(NATIVE_FILE.read_bytes()).hexdigest()}
