"""Execution-only adapters for the confirmed v1.1 protocol; architecture unchanged."""
import hashlib
import itertools
import torch
from codex.sid_order_launch.sid_order_shuffle_model import SemanticIDEncoderDecoder as OriginalShuffle
from codex.sid_order_launch.sid_order_mlp_model import SemanticIDEncoderDecoder as OriginalMLP
from src.models.components.interfaces import OneKeyPerPredictionOutput


def user_ids(model_input):
    ids = model_input.transformed_sequences['user_id']
    return [int(v) for v in ids[:, 0].detach().cpu().tolist()]


class ProtocolMixin:
    def __init__(self, *args, dataset_name, eval_seed=2026, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.eval_seed = eval_seed
        self.shuffle_calls = 0
        self.is_testing = False
        self.save_attentions_on_test = False
        self.save_attentions_on_predict = False
        self.protocol_sample_ids = set()
        self.protocol_references = {}
        self._permutation_cache = {}

    def on_test_start(self):
        super().on_test_start()
        self.is_testing = True

    def on_test_end(self):
        super().on_test_end()
        self.is_testing = False

    def eval_step(self, batch, loss_to_aggregate, **kwargs):
        # Full metric evaluation deliberately never requests attention tensors.
        inputs, labels = batch
        _, loss = self.model_step(model_input=inputs, label_data=labels)
        ids, scores = self.generate(attention_mask=inputs.mask, output_attentions=False, **{
            self.feature_to_model_input_map.get(k, k): v
            for k, v in inputs.transformed_sequences.items()})
        self.evaluator(marginal_probs=scores, generated_ids=ids,
                       labels=list(labels.labels.values())[0].to(scores.device))
        loss_to_aggregate(loss)
        if self.is_testing:
            sink = getattr(self, 'normal_candidate_sink', None)
            if sink is not None:
                sink(inputs, labels, ids, scores)
            for i, uid in enumerate(user_ids(inputs)):
                if uid in self.protocol_sample_ids and uid not in self.protocol_references:
                    self.protocol_references[uid] = {
                        'ids': ids[i].detach().cpu(), 'scores': scores[i].detach().cpu()}

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, self.test_loss)

    def predict_step(self, batch, batch_idx=0):
        # Avoid original MLP predict_step's unconditional shared-directory export.
        ids, scores = self.generate(attention_mask=batch.mask, output_attentions=False, **{
            self.feature_to_model_input_map.get(k, k): v
            for k, v in batch.transformed_sequences.items()})
        return OneKeyPerPredictionOutput(keys=batch.user_id_list, predictions=ids)

    def encoder_forward_pass(self, attention_mask, input_ids, user_id, output_attentions=False):
        self._protocol_user_ids = [int(x) for x in user_id[:, 0].detach().cpu().tolist()]
        return super().encoder_forward_pass(attention_mask=attention_mask, input_ids=input_ids,
            user_id=user_id, output_attentions=output_attentions)

    def permutation(self, uid, num_items, count):
        key = (uid, num_items, count)
        if key not in self._permutation_cache:
            choices = list(itertools.permutations(range(count)))
            rows = []
            for pos in range(num_items):
                token = f'{self.eval_seed}|{self.dataset_name}|{uid}|{pos}|shuffle_{count}'.encode()
                # Local hash-based selection consumes no train/global RNG state.
                number = int.from_bytes(hashlib.sha256(token).digest(), 'big')
                rows.append(list(choices[number % len(choices)]) + list(range(count, self.num_hierarchies)))
            self._permutation_cache[key] = torch.tensor(rows, dtype=torch.long)
        return self._permutation_cache[key]

    def _shuffle_hierarchical_embeddings(self, id_embeddings, attention_mask,
                                         num_hierarchies=4, hierarchies_to_shuffle=3):
        assert self.is_testing and not self.training, 'Shuffle prohibited in training/validation'
        self.shuffle_calls += 1
        b, length, dim = id_embeddings.shape
        n = length // num_hierarchies
        indices = torch.stack([self.permutation(uid, n, hierarchies_to_shuffle)
                               for uid in self._protocol_user_ids]).to(id_embeddings.device)
        valid = attention_mask.reshape(b, n, num_hierarchies).any(-1)
        identity = torch.arange(num_hierarchies, device=indices.device).view(1, 1, -1)
        indices = torch.where(valid.unsqueeze(-1), indices, identity)
        self.protocol_last_permutation = indices.detach().cpu()
        shaped = id_embeddings.reshape(b, n, num_hierarchies, dim)
        return shaped.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, dim)).reshape(b, length, dim)


class ShuffleModel(ProtocolMixin, OriginalShuffle):
    pass


class MLPModel(ProtocolMixin, OriginalMLP):
    pass
