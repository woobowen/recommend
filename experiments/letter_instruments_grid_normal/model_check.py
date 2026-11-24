"""CPU fixture for the actual model instance and BOS/SEP wiring."""
import argparse
import json
import os
from pathlib import Path
import sys

from data_assets import official
from run_normal import configure, model_identity, NATIVE_ROOT


def check(assets, output):
    sys.path.insert(0, str(NATIVE_ROOT))
    import lightning as L
    import torch
    from hydra.utils import instantiate

    cfg, _ = configure(assets, output, False)
    codes, seqs, _ = official(assets)
    L.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("medium")
    model = instantiate(cfg.model).eval()
    summary = model_identity(model, codes)
    history = [v for item in seqs[0][-21:-1] for v in codes[item]]
    input_ids = torch.tensor([history + [-1] * (120 - len(history))])
    mask = input_ids.ne(-1).long()
    user = torch.tensor([[0]])
    observations = {}

    def sep_probe(_module, _args, kwargs):
        embeddings = kwargs["sequence_embedding"]
        enc_mask = kwargs["attention_mask"]
        assert embeddings.shape == (1, 150, 128) and enc_mask.shape == (1, 150)
        assert torch.equal(embeddings[0, 4::5], model.sep_token.expand(30, -1))
        assert enc_mask.sum().item() == len(history) * 5 // 4
        assert enc_mask[0].tolist() == [1] * (len(history) * 5 // 4) + [0] * (150 - len(history) * 5 // 4)
        valid = input_ids[0, :len(history)] + (torch.arange(len(history)) % 4) * 256
        actual = embeddings.reshape(1, 30, 5, 128)[0, :, :4].reshape(120, 128)
        assert torch.equal(actual[:len(history)], model.get_embedding_table("encoder")(valid))
        observations["sep_and_offsets"] = True

    def bos_probe(_module, _args, kwargs):
        embeddings = kwargs["sequence_embedding"]
        assert embeddings.shape == (1, 1, 128)
        assert torch.equal(embeddings[0], model.decoder.bos_token)
        observations["starts_with_learned_bos"] = True

    h1 = model.encoder.register_forward_pre_hook(sep_probe, with_kwargs=True)
    h2 = model.decoder.register_forward_pre_hook(bos_probe, with_kwargs=True)
    try:
        with torch.no_grad():
            encoded, enc_mask = model.encoder_forward_pass(mask, input_ids, user)
            decoded = model.decoder_forward_pass(encoder_output=encoded, attention_mask_for_encoder=enc_mask)
            assert decoded.shape == (1, 1, 128) and torch.isfinite(decoded).all()
    finally:
        h1.remove()
        h2.remove()
    assert len(observations) == 2
    return {"model": summary, "fixture": observations,
            "visible_sid_tokens": len(history), "encoder_visible_with_sep": len(history) * 5 // 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(json.dumps(check(args.assets, args.output), indent=2))
