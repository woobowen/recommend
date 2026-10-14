"""Bounded semantic probes used by the saved preflight runner.

These are engineering probes, never formal model initialization or metrics.
They accept an already constructed model; they do not select assets, training
conditions, or checkpoint migration policies.
"""

from __future__ import annotations

import torch

from .models import AuthorMLPModel, AuthorShuffleModel, OriginalShuffleModel, validate_item_groups


def describe_author_model(model) -> dict:
    encoder_table = model.get_embedding_table("encoder")
    decoder_table = model.get_embedding_table("decoder")
    same_table = encoder_table is decoder_table
    if isinstance(model, AuthorShuffleModel) and same_table:
        raise AssertionError("Author route A must preserve independent SID tables")
    if isinstance(model, AuthorMLPModel):
        if not same_table or model.sep_token is not None:
            raise AssertionError("Author MLP must preserve its shared SID table and no SEP")
        layers = list(model.item_aggregation_mlp)
        expected_types = [torch.nn.Linear, torch.nn.ReLU, torch.nn.Dropout, torch.nn.Linear]
        if [type(layer) for layer in layers] != expected_types:
            raise AssertionError("Author per-item aggregation structure changed")
        e, h = model.embedding_dim, model.num_hierarchies
        if (layers[0].in_features, layers[0].out_features,
            layers[3].in_features, layers[3].out_features) != (h * e, 2 * e, 2 * e, e):
            raise AssertionError("Author per-item aggregation dimensions changed")
        if layers[2].p != 0.1 or layers[0].bias is None or layers[3].bias is None:
            raise AssertionError("Author aggregation Dropout/bias contract changed")
    return {
        "adapter_class": type(model).__name__,
        "num_hierarchies": model.num_hierarchies,
        "num_embeddings_per_hierarchy": model.num_embeddings_per_hierarchy,
        "embedding_dim": model.embedding_dim,
        "top_k_for_generation": model.top_k_for_generation,
        "shared_sid_table": same_table,
        "encoder_sid_table_shape": list(encoder_table.weight.shape),
        "decoder_sid_table_shape": list(decoder_table.weight.shape),
        "has_sep": model.sep_token is not None,
        "has_user_table": model.user_embedding is not None,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
        "state_dict_shapes": {key: list(value.shape) for key, value in model.state_dict().items()},
    }


def check_item_aggregation(model: AuthorMLPModel, input_ids, attention_mask) -> dict:
    """Probe the real looked-up history while controlling only aggregator Dropout."""
    if not isinstance(model, AuthorMLPModel):
        raise TypeError("This probe requires AuthorMLPModel")
    validate_item_groups(attention_mask, model.num_hierarchies)
    aggregator = model.item_aggregation_mlp
    was_training = aggregator.training
    aggregator.eval()
    try:
        with torch.no_grad():
            shifted = model._add_repeating_offset_to_rows(
                input_ids, model.num_embeddings_per_hierarchy, model.num_hierarchies,
                attention_mask=attention_mask,
            )
            embeds = model.get_embedding_table("encoder")(shifted)
            b, tokens, e = embeds.shape
            m, h = tokens // model.num_hierarchies, model.num_hierarchies
            if b < 1 or m < 1:
                raise ValueError("Aggregation preflight needs at least one real history item")
            flattened = embeds.reshape(b, m, h * e)
            aggregated = aggregator(flattened)
            if tuple(aggregated.shape) != (b, m, e):
                raise AssertionError("Aggregation collapsed or expanded the item axis")
            # An explicit local perturbation verifies that the shared MLP acts
            # independently before the later encoder performs history attention.
            perturbed = flattened.clone()
            perturbed[0, 0, 0] += 0.125
            changed = aggregator(perturbed)
            unaffected = torch.ones((b, m), dtype=torch.bool, device=flattened.device)
            unaffected[0, 0] = False
            if not torch.equal(changed[unaffected], aggregated[unaffected]):
                raise AssertionError("An item changed another item in the aggregation stage")
            variable_shapes = []
            for bb, mm in {(1, 1), (b, min(m, 2)), (b, m)}:
                output = aggregator(flattened[:bb, :mm])
                if tuple(output.shape) != (bb, mm, e):
                    raise AssertionError("Aggregator depends on fixed B or M")
                variable_shapes.append([bb, mm, e])
            return {
                "status": "PASS", "probe_only": True,
                "input_lookup_shape": [b, m, h, e],
                "mlp_input_shape": list(flattened.shape),
                "mlp_output_shape": list(aggregated.shape),
                "unaffected_items_equal": True,
                "variable_output_shapes": sorted(variable_shapes),
                "mask_rule": "last SID code of each item; complete-group contract checked",
                "dropout_for_independence_check": "eval only; prior state restored",
            }
    finally:
        aggregator.train(was_training)


def check_original_shuffle_trace(model: AuthorShuffleModel) -> dict:
    """Compare raw original and bridge, including actual global RNG consumption.

    This probe saves/restores RNG and trace state intentionally. It must run only
    as a preflight check, never in the middle of a formal randomization sequence.
    """
    if not isinstance(model, AuthorShuffleModel):
        raise TypeError("This probe requires AuthorShuffleModel")
    device = model.get_embedding_table("encoder").weight.device
    get_state = (lambda: torch.cuda.get_rng_state(device)) if device.type == "cuda" else torch.get_rng_state
    set_state = (lambda state: torch.cuda.set_rng_state(state, device)) if device.type == "cuda" else torch.set_rng_state
    initial_state = get_state()
    old_events, old_limit = model.permutation_events, model.permutation_trace_limit
    old_replay = model._replay_queue
    old_active, old_enabled = model.is_testing, model.shuffle_embedding_on_test
    report = []
    try:
        h, b, m, e = model.num_hierarchies, 2, 3, 5
        embeds = torch.arange(b * m * h * e, device=device, dtype=torch.float32).reshape(b, m * h, e)
        original_embeds = embeds.clone()
        mask = torch.ones(b, m, h, device=device, dtype=torch.bool)
        mask[0, -1] = False
        mask = mask.reshape(b, m * h)
        for n in (3, 4):
            if n > h:
                continue
            set_state(initial_state)
            expected = OriginalShuffleModel._shuffle_hierarchical_embeddings(
                model, embeds.clone(), mask, h, n,
            )
            raw_final_state = get_state()
            set_state(initial_state)
            model.permutation_events = []
            model.permutation_trace_limit = 1
            model._replay_queue = None
            observed = model._shuffle_hierarchical_embeddings(embeds, mask, h, n)
            bridge_final_state = get_state()
            if not torch.equal(observed, expected) or not torch.equal(raw_final_state, bridge_final_state):
                raise AssertionError("Bridge differs from original output or global RNG consumption")
            if not torch.equal(embeds, original_embeds):
                raise AssertionError("Bridge mutated the caller's OFF input")
            permutation = torch.tensor(model.permutation_events[0]["permutation"], device=device)
            grouped_before, grouped_after = embeds.reshape(b, m, h, e), observed.reshape(b, m, h, e)
            for bi in range(b):
                for mi in range(m):
                    if not mask.reshape(b, m, h)[bi, mi, -1]:
                        if not torch.equal(grouped_after[bi, mi], grouped_before[bi, mi]):
                            raise AssertionError("Padding item changed")
                        continue
                    for position in range(n):
                        source = int(permutation[bi, mi, position].item())
                        if not torch.equal(grouped_after[bi, mi, position], grouped_before[bi, mi, source]):
                            raise AssertionError("Trace does not identify the original per-item vector")
            if not torch.equal(grouped_after[:, :, n:], grouped_before[:, :, n:]):
                raise AssertionError("Unshuffled code positions changed")
            # Guarantee a nonidentity replay example while retaining the random
            # helper's unrestricted (including identity) formal distribution.
            model.is_testing, model.shuffle_embedding_on_test = True, True
            nonidentity = torch.arange(n, device=device).roll(1).expand(b, m, n).clone()
            nonidentity[0, 1] = torch.arange(n, device=device).flip(0)
            model.permutation_events = []
            with model.replay_permutations([nonidentity]):
                replayed = model._shuffle_hierarchical_embeddings(embeds, mask, h, n)
            for position in range(n):
                if not torch.equal(replayed.reshape(b, m, h, e)[1, 0, position],
                                   grouped_before[1, 0, int(nonidentity[1, 0, position])]):
                    raise AssertionError("Controlled nonidentity replay failed")
            report.append({
                "n": n, "original_equals_adapter": True,
                "global_rng_equals_original": True, "caller_input_unchanged": True,
                "padding_unchanged": True, "unshuffled_codes_unchanged": True,
                "trace_matches_moved_vectors": True,
                "random_permutation": permutation.cpu().tolist(),
                "nonidentity_replay": nonidentity.cpu().tolist(),
            })
        return {"status": "PASS", "probe_only": True, "checks": report}
    finally:
        set_state(initial_state)
        model.permutation_events, model.permutation_trace_limit = old_events, old_limit
        model._replay_queue = old_replay
        model.is_testing, model.shuffle_embedding_on_test = old_active, old_enabled
