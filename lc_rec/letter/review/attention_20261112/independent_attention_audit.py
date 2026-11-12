#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, datetime, hashlib, json, math, sys
from pathlib import Path
import torch
torch.set_num_threads(4)
ATOL, RTOL, CHUNK = 1e-6, 1e-3, 1 << 20

def digest(t):
    raw = t.detach().contiguous().reshape(-1).view(torch.uint8)
    h = hashlib.sha256()
    for i in range(0, raw.numel(), 8 << 20):
        h.update(memoryview(raw[i:i + (8 << 20)].numpy()))
    return h.hexdigest()

def load(path):
    t = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    assert isinstance(t, torch.Tensor) and t.device.type == "cpu"
    assert t.layout == torch.strided
    return t

def scalar(x):
    x = float(x)
    return x if math.isfinite(x) else repr(x)

def diff(a, b):
    out = {"left_shape": list(a.shape), "right_shape": list(b.shape),
           "left_dtype": str(a.dtype), "right_dtype": str(b.dtype)}
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(out, compatible=False, exact=False, tolerance_pass=False)
    a, b = a.detach().contiguous().reshape(-1), b.detach().contiguous().reshape(-1)
    ha, hb = digest(a), digest(b)
    out.update(compatible=True, elements=a.numel(), left_sha256=ha, right_sha256=hb, exact=ha == hb)
    different = exceeded = nonfinite = 0
    maxabs = maxexcess = 0.0
    first = []
    floating = torch.is_floating_point(a)
    for i in range(0, a.numel(), CHUNK):
        x, y = a[i:i + CHUNK], b[i:i + CHUNK]
        neq = x != y
        different += int(neq.sum())
        u, v = x.to(torch.float64), y.to(torch.float64)
        delta, limit = (v - u).abs(), ATOL + RTOL * u.abs()
        finite = torch.isfinite(u) & torch.isfinite(v)
        nonfinite += int((~finite).sum())
        bad = ((~finite) | (delta > limit)) if floating else neq
        exceeded += int(bad.sum())
        if bool(finite.any()):
            maxabs = max(maxabs, float(delta[finite].max()))
            maxexcess = max(maxexcess, float((delta[finite] - limit[finite]).clamp_min(0).max()))
        positions = torch.nonzero(neq, as_tuple=False).reshape(-1)
        for j in positions[:max(0, 8 - len(first))].tolist():
            first.append({"index": i + j, "left": scalar(u[j]), "right": scalar(v[j]),
                          "abs_diff": scalar(delta[j]), "limit": scalar(limit[j])})
    out.update(different_elements=different, nonfinite_elements=nonfinite,
               max_abs_diff=maxabs if not nonfinite else "nonfinite",
               tolerance_exceeded_elements=exceeded, max_tolerance_excess=maxexcess,
               tolerance_pass=(exceeded == 0) if floating else ha == hb, first8diff=first)
    return out

def inventory(directory, rank):
    paths = {p.stem: p for p in directory.glob("*.pt")}
    sidecars = {p.stem for p in directory.glob("*.json")}
    result = {"files": {}, "errors": [], "tensor_count": len(paths)}
    if len(paths) != 88:
        result["errors"].append(f"expected 88 tensors, found {len(paths)}")
    if set(paths) != sidecars:
        result["errors"].append("Tensor/JSON filename sets differ")
    for key, path in sorted(paths.items()):
        try:
            m, t = json.loads(path.with_suffix(".json").read_text()), load(path)
            assert m["shape"] == list(t.shape) and m["dtype"] == str(t.dtype)
            assert m["sha256"] == digest(t) and m["elements"] == t.numel()
            assert (m["rank"], m["layer"], m["step"], m["micro"]) == (rank, 28, 2, 9)
            assert m["phase"] in ("primary", "recompute") and key.startswith(m["phase"] + ".")
            assert len(m["stride"]) == t.ndim
            result["files"][key] = {"path": str(path), "metadata": m,
                "independent_norm": scalar(t.float().norm()),
                "independent_max_abs": scalar(t.float().abs().max()) if t.numel() else 0}
        except Exception as e:
            result["errors"].append(f"{key}: {type(e).__name__}: {e}")
    return result

def mathematics(inv):
    checks = []
    def t(label):
        return load(inv["files"]["recompute." + label]["path"])
    def check(name, fn):
        try:
            a, b, mapping = fn()
            checks.append({"name": name, "mapping": mapping, "comparison": diff(a, b)})
        except Exception as e:
            checks.append({"name": name, "error": f"{type(e).__name__}: {e}"})
    for k in ("q", "k", "v", "out", "lse", "dO", "dQ"):
        check("native_dispatch_" + k, lambda k=k: (t("native." + k), t("dispatch." + k), "identity"))
    for k, index in (("K", 0), ("V", 1)):
        check("packed_d" + k, lambda k=k, index=index:
              (t("native.dKV")[:, index], t("dispatch.d" + k), f"dKV[:,{index}]"))
    check("Q_gradient", lambda: (t("native.dQ"), t("unpad.q.post.grad"), "identity"))
    check("KV_gradient", lambda: (t("native.dKV"), t("unpad.kv.post.grad"), "identity"))
    check("Q_input", lambda: (t("native.q"), t("unpad.q.post"), "identity"))
    check("K_input", lambda: (t("native.k"), t("unpad.kv.post")[:, 0], "KV[:,0]"))
    check("V_input", lambda: (t("native.v"), t("unpad.kv.post")[:, 1], "KV[:,1]"))
    for branch in ("q", "kv"):
        def scatter(branch=branch):
            pre, post, idx = t(f"unpad.{branch}.pre.grad"), t(f"unpad.{branch}.post.grad"), t(f"unpad.{branch}.indices")
            assert idx.ndim == 1 and idx.dtype in (torch.int32, torch.int64)
            idx = idx.long()
            assert idx.unique().numel() == idx.numel()
            rows = pre.shape[0] * pre.shape[1]
            assert not idx.numel() or (int(idx.min()) >= 0 and int(idx.max()) < rows)
            assert tuple(post.shape) == (idx.numel(),) + tuple(pre.shape[2:])
            rebuilt = torch.zeros_like(pre).reshape((rows,) + tuple(pre.shape[2:]))
            rebuilt.index_copy_(0, idx, post)
            return rebuilt.reshape(pre.shape), pre, "unique-index zero-filled scatter"
        check("unpad_" + branch + "_scatter", scatter)
    def qmap():
        projection = t("q_proj.output.grad")
        rotary = t("rope.q.pre.grad")
        assert rotary.ndim == 4 and projection.ndim == 3
        assert rotary.shape[:2] == projection.shape[:2]
        assert rotary.shape[2] * rotary.shape[3] == projection.shape[2] == 4096
        return rotary.reshape(projection.shape), projection, "BSHD -> BSC"
    check("rope_pre_to_q_proj_gradient", qmap)
    def qpost():
        rotary, unpad = t("rope.q.post.grad"), t("unpad.q.pre.grad")
        assert rotary.shape == unpad.shape
        return rotary, unpad, "identity BSHD"
    check("rope_post_to_unpad_pre_gradient", qpost)
    return checks

def events(training, rank):
    rows = [json.loads(s) for s in (training / f"events.rank{rank}.jsonl").read_text().splitlines() if s.strip()]
    out = {"errors": []}
    try:
        for name in ("fa_backward_ctx", "fa_native_dispatch"):
            selected = [r for r in rows if r["event"] == name]
            assert len(selected) == 1, (name, len(selected))
            r = selected[0]
            assert (r["rank"], r["layer"], r["step"], r["micro"], r["phase"]) == (rank, 28, 2, 9, "recompute")
            assert isinstance(r["deterministic"], bool)
            out[name] = r
        assert out["fa_backward_ctx"]["deterministic"] == out["fa_native_dispatch"]["deterministic"]
        calls = [r for r in rows if r["event"] == "fa_effective_call"]
        assert sorted(r["phase"] for r in calls) == ["primary", "recompute"]
        out["fa_effective_call"] = calls
        batches = [r for r in rows if r["event"] == "batch"]
        assert [r["loader_yielded"] for r in batches] == list(range(9, 17))
        assert [r["step"] for r in batches] == [2] * 4 + [3] * 4
        updates = [r for r in rows if r["event"] == "optimizer_step"]
        assert [r["step"] for r in updates] == [3, 4]
        assert all(r["lr"] and all(math.isfinite(float(v)) and float(v) > 0 for v in r["lr"]) for r in updates)
        out.update(batch_count=len(batches), optimizer_steps=updates,
                   lr_scope="Logged optimizer_step LR; actual Adam LR requires separate S4 evidence")
    except Exception as e:
        out["errors"].append(f"{type(e).__name__}: {e}")
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pair", action="append", required=True)
    args = p.parse_args()
    pairs = [x.split(",") for x in args.pair]
    assert all(len(x) == 2 for x in pairs)
    jobs = sorted({j for pair in pairs for j in pair})
    assert all("/" not in j and j not in (".", "..") for j in jobs)
    report = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "argv": sys.argv, "torch": torch.__version__, "CUDA_VISIBLE_DEVICES": "",
              "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "tolerance": {"atol": ATOL, "rtol": RTOL, "reference": "left"}, "jobs": {}, "pairs": []}
    for job in jobs:
        training = Path(args.root) / job / "training"
        report["jobs"][job] = {}
        for rank in (0, 1):
            inv = inventory(training / f"boundary.rank{rank}", rank)
            report["jobs"][job][str(rank)] = {"inventory": inv, "mathematics": mathematics(inv), "events": events(training, rank)}
    for left, right in pairs:
        item = {"left": left, "right": right, "ranks": {}}
        for rank in ("0", "1"):
            a, b = (report["jobs"][j][rank]["inventory"]["files"] for j in (left, right))
            rows = {k: diff(load(a[k]["path"]), load(b[k]["path"])) for k in sorted(a.keys() & b.keys())}
            for k, row in rows.items():
                row["original_metadata_equal"] = {f: a[k]["metadata"][f] == b[k]["metadata"][f]
                    for f in ("shape", "dtype", "stride", "layout", "storage_offset")}
            item["ranks"][rank] = {"only_left": sorted(a.keys() - b.keys()), "only_right": sorted(b.keys() - a.keys()), "tensors": rows}
        report["pairs"].append(item)
    report["limitation"] = "CPU bridge audit; no acceptance-state, profiler, independent GPU rerun, or overall recovery-gate verdict."
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))

if __name__ == "__main__":
    main()
