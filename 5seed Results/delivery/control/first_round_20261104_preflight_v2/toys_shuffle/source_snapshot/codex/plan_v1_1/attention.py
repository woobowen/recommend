"""Freeze 100 users before outcomes; export final-Top1-prefix raw attention."""
import fcntl
import gzip
import hashlib
import json
import random
from pathlib import Path
import torch
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.components.interfaces import SequentialModelInputData, SequentialModuleLabelData
from codex.plan_v1_1.protocol import user_ids


def get_test_loader(pipeline):
    dm = pipeline.datamodule
    dm.trainer = pipeline.trainer
    dm.setup('test')
    loader = dm.test_dataloader()
    return loader[0] if isinstance(loader, (tuple, list)) else loader


def prepare_samples(pipeline, directory, dataset, input_hashes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / f'{dataset}_100_users.json'
    data_path = directory / f'{dataset}_100_users.pt'
    with (directory / f'{dataset}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            assert manifest['input_hashes'] == input_hashes
            assert hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest['samples_sha256']
            return torch.load(data_path, weights_only=False), manifest
        unique = {}
        ordinal = 0
        for inputs, labels in get_test_loader(pipeline):
            size = inputs.mask.shape[0]
            reshaped_labels = {k: v.reshape(size, -1) for k, v in labels.labels.items()}
            for i, uid in enumerate(user_ids(inputs)):
                ordinal += 1
                length = int(inputs.mask[i].sum().item()) // 4
                valid = length > 0 and all(v.shape[1] == 4 and bool(((v[i] >= 0) & (v[i] < 256)).all())
                                          for v in reshaped_labels.values())
                if uid in unique or not valid:
                    continue
                sample_input = SequentialModelInputData(user_id_list=[uid], mask=inputs.mask[i:i+1].clone(),
                    transformed_sequences={k: v[i:i+1].clone() for k, v in inputs.transformed_sequences.items()})
                sample_label = SequentialModuleLabelData(labels={k: v[i:i+1].clone() for k, v in reshaped_labels.items()})
                unique[uid] = {'user_id': uid, 'history_items': length, 'record_ordinal': ordinal,
                               'input': sample_input, 'labels': sample_label}
        ordered = sorted(unique.values(), key=lambda s: (s['history_items'], s['user_id']))
        assert len(ordered) >= 100
        n = len(ordered)
        groups = [ordered[:n//3], ordered[n//3:2*n//3], ordered[2*n//3:]]
        rng = random.Random(2026)
        selected = []
        for group, count in zip(groups, [34, 33, 33]):
            selected.extend(rng.sample(group, min(count, len(group))))
        if len(selected) < 100:
            chosen = {s['user_id'] for s in selected}
            selected.extend(rng.sample([s for s in ordered if s['user_id'] not in chosen], 100-len(selected)))
        torch.save(selected, data_path)
        manifest = {'dataset': dataset, 'sample_seed': 2026, 'valid_unique_users': n,
            'input_hashes': input_hashes, 'selection': 'length,user_id sorted thirds; 34/33/33; first valid record',
            'samples_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
            'users': [{k: s[k] for k in ('user_id', 'history_items', 'record_ordinal')} for s in selected]}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        return selected, manifest


def export_attention(model, samples, directory, metadata, references, require_references=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.eval()
    model._make_deterministic(False)
    model.is_testing = True
    device = model.device
    exported = []
    with torch.inference_mode():
        for sample in samples:
            raw = sample['input']
            mask = raw.mask.to(device)
            kwargs = {model.feature_to_model_input_map.get(k, k): v.to(device)
                      for k, v in raw.transformed_sequences.items()}
            uid = sample['user_id']
            ids, scores = model.generate(attention_mask=mask, output_attentions=False, **kwargs)
            if require_references:
                assert uid in references, f'Missing full-test reference for {uid}'
            if uid in references:
                assert torch.equal(ids[0].cpu(), references[uid]['ids']), f'Full/subset prediction mismatch {uid}'
                torch.testing.assert_close(scores[0].cpu(), references[uid]['scores'], rtol=1e-4, atol=1e-6)
            # Check that requesting attention does not alter the generated ranking.
            with_attention = model.generate(attention_mask=mask, output_attentions=True, **kwargs)
            assert torch.equal(ids, with_attention[0])
            torch.testing.assert_close(scores, with_attention[1], rtol=1e-5, atol=1e-7)
            del with_attention
            # Recompute a coherent single path: BOS, Top1 H0, H1, H2 -> four query rows.
            encoder, encoder_mask, encoder_attention = model.encoder_forward_pass(
                attention_mask=mask, output_attentions=True, **kwargs)
            top1 = ids[:, 0, :]
            output, decoder_self, decoder_cross = model.decoder_forward_pass(
                future_ids=top1[:, :-1], encoder_output=encoder,
                attention_mask_for_encoder=encoder_mask, use_cache=False, output_attentions=True)
            arrays = {'encoder_self': tuple(t.detach().cpu() for t in encoder_attention),
                      'decoder_self': tuple(t.detach().cpu() for t in decoder_self),
                      'decoder_cross': tuple(t.detach().cpu() for t in decoder_cross)}
            assert all(len(v) == 4 for v in arrays.values())
            assert all(t.shape[1] == 6 and torch.isfinite(t).all() for v in arrays.values() for t in v)
            assert arrays['decoder_cross'][-1].shape[2] == 4
            n = raw.mask.shape[1] // 4
            identity = torch.arange(4).expand(1, n, 4).clone()
            permutation = getattr(model, 'protocol_last_permutation', identity) if metadata['mode'].startswith('shuffle_') else identity
            payload = dict(metadata, user_id=uid, record_ordinal=sample['record_ordinal'],
                input_sids=raw.transformed_sequences['sequence_data'], input_mask=raw.mask,
                encoder_mask=encoder_mask.cpu(), history_items=sample['history_items'],
                item_position_indices=torch.arange(n), sid_slots_per_item=4,
                encoder_tokens_per_item=1 if metadata['variant']=='mlp' else (5 if model.sep_token is not None else 4),
                sep_after_sid=model.sep_token is not None, permutation=permutation,
                true_labels=sample['labels'].labels, top10_ids=ids.cpu(), top10_scores=scores.cpu(),
                top1_prefix=top1[:, :-1].cpu(), decoder_query_labels=['H0','H1','H2','H3'],
                decoder_key_labels=['BOS','H0','H1','H2'], attention=arrays,
                cross_last_layer_head_mean=arrays['decoder_cross'][-1].mean(1))
            path = directory / f'user_{uid}.pt.gz'
            with gzip.open(path, 'wb') as stream:
                torch.save(payload, stream)
            with gzip.open(path, 'rb') as stream:
                reloaded = torch.load(stream, weights_only=False)
            assert reloaded['user_id'] == uid
            exported.append({'user_id': uid, 'file': path.name,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    result = dict(metadata, users=len(exported), files=exported,
                  attention_switch_prediction_check=True, full_test_subset_check=require_references)
    (directory / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    model.is_testing = False
    return result
