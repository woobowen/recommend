"""Shared identity, configuration relocation and real data utilities."""
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
OLD_QWEN = "/home/jcyin/GRID_wmh_Qwen_TIGER_20261009"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_asset(record):
    p = Path(record["path"])
    if not p.is_file():
        raise ValueError("Missing real asset: " + str(p))
    with p.open("rb") as stream:
        if stream.read(48).startswith(b"version https://git-lfs"):
            raise ValueError("LFS pointer, not an asset: " + str(p))
    if sha256(p) != record["sha256"]:
        raise ValueError("Asset hash mismatch: " + str(p))
    return {**record, "bytes": p.stat().st_size, "resolved": str(p.resolve())}


def bind_source(route):
    root = REPO / ("GRID baseline" if route == "B" else "references/wmh-original/GRID")
    if "src" in sys.modules:
        actual = Path(sys.modules["src"].__file__).resolve()
        if root.resolve() not in actual.parents:
            raise RuntimeError("Different src namespace already loaded: " + str(actual))
    sys.path.insert(0, str(root))
    os.chdir(root)
    return root


def load_config(spec, output):
    from omegaconf import OmegaConf
    import src.utils.custom_hydra_resolvers  # registers project resolver names
    root = bind_source(spec["route"])
    verify_asset(spec["config"])
    text = Path(spec["config"]["path"]).read_text()
    cfg = OmegaConf.create(text)
    # Explicit paths only; original data/checkpoint contents remain read only.
    cfg.data_dir = spec["data_dir"]
    cfg.semantic_id_path = spec["sid"]["path"]
    cfg.ckpt_path = spec.get("checkpoint", {}).get("path")
    for key, value in {"root_dir": root, "work_dir": root, "output_dir": output,
                       "log_dir": output, "profile_dir": output / "profile_output",
                       "metadata_dir": output / "metadata"}.items():
        cfg.paths[key] = str(value)
    cfg.seed = spec["seed"]
    cfg.train = False
    cfg.test = True
    if spec["route"] == "B":
        cfg.model._target_ = "experiments.beauty_extensions.baseline.BaselineShuffleModel"
        cfg.model.shuffle_n = spec.get("shuffle_n", 0)
    else:
        target = "AuthorShuffleModel" if spec["route"] == "A" else "AuthorMLPModel"
        cfg.model._target_ = "experiments.beauty_extensions.author.models." + target
        if spec["route"] == "A":
            cfg.model.shuffle_embedding_on_test = spec.get("shuffle_n", 0) != 0
            cfg.model.hierarchies_to_shuffle = spec.get("shuffle_n", 0) or 3
        cfg.model.save_attentions_on_test = False
    if "generation_k" in spec:
        cfg.model.top_k_for_generation = spec["generation_k"]
    for split in ("train", "val", "test"):
        loader = cfg.data_loading[f"{split}_dataloader_config"].dataloader
        loader.data_folder = str(Path(spec["data_dir"]) / {"train": "training", "val": "evaluation", "test": "testing"}[split])
        if not spec.get("formal_approved", False):
            loader.num_workers = 0
            loader.timeout = 0
            loader.persistent_workers = False
        # Only the known SID loader is instantiated. Template placeholders stay.
        sid_loader = loader.dataset_config.semantic_id_map.sequence_data
        sid_loader._args_[0].file_path = spec["sid"]["path"]
        sid_loader.weights_only = True
    cfg.callbacks.restart_job = None
    cfg.trainer.devices = 1
    cfg.trainer.strategy = "auto"
    cfg.trainer.enable_progress_bar = False
    return cfg


def load_checkpoint(path):
    import builtins
    import importlib
    import torch
    from .checkpoint_info import globals_in_checkpoint
    # Explicit names statically inspected in the hash-verified working checkpoint.
    # The context is local to this single load; unexpected globals fail closed.
    approved = {
        "__builtin__ dict", "__builtin__ list", "__builtin__ long", "__builtin__ set",
        "collections OrderedDict", "collections defaultdict", "functools partial",
        "omegaconf.base ContainerMetadata", "omegaconf.base Metadata",
        "omegaconf.dictconfig DictConfig", "omegaconf.listconfig ListConfig", "omegaconf.nodes AnyNode",
        "src.components.eval_metrics NDCG", "src.components.eval_metrics Recall",
        "src.components.eval_metrics SIDRetrievalEvaluator",
        "src.data.loading.components.collate_functions collate_fn_train",
        "src.data.loading.components.collate_functions collate_with_sid_causal_duplicate",
        "src.data.loading.components.dataloading UnboundedSequenceIterable",
        "src.data.loading.components.interfaces SemanticIDDatasetConfig",
        "src.data.loading.components.interfaces SequenceDataloaderConfig",
        "src.data.loading.components.iterators TFRecordIterator",
        "src.data.loading.components.label_function NextKTokenMasking",
        "src.data.loading.components.pre_processing convert_fields_to_tensors",
        "src.data.loading.components.pre_processing convert_to_dense_numpy_array",
        "src.data.loading.components.pre_processing filter_features_to_consider",
        "src.data.loading.components.pre_processing map_sparse_id_to_semantic_id",
        "tensorflow.python.framework.dtypes as_dtype", "tensorflow.python.ops.parsing_config VarLenFeature",
        "torch FloatStorage", "torch LongStorage", "torch device", "torch._utils _rebuild_tensor_v2",
        "torch.optim.adam Adam", "torchmetrics.metric jit_distributed_available", "typing Any",
    }
    names = globals_in_checkpoint(path)["global_names"]
    if set(names) - approved:
        raise ValueError("Unreviewed checkpoint globals: " + repr(sorted(set(names) - approved)))
    objects = []
    for name in names:
        module, attribute = name.split(" ", 1)
        if module == "__builtin__":
            obj = getattr(builtins, "int" if attribute == "long" else attribute)
        else:
            obj = getattr(importlib.import_module(module), attribute)
        objects.append(obj)
    with torch.serialization.safe_globals(objects):
        return torch.load(path, map_location="cpu", weights_only=True)


def instantiate(spec, output, checkpoint=True):
    import hydra
    import torch
    from lightning import seed_everything
    from omegaconf import OmegaConf
    cfg = load_config(spec, output)
    verified = [verify_asset(spec["sid"])]
    if checkpoint and spec.get("checkpoint"):
        verified.append(verify_asset(spec["checkpoint"]))
    seed_everything(spec["seed"], workers=True)
    torch.set_float32_matmul_precision("medium")  # saved working entrypoint protocol
    model = hydra.utils.instantiate(cfg.model)
    strict = None
    if checkpoint and spec.get("checkpoint"):
        state = load_checkpoint(spec["checkpoint"]["path"])
        loaded = model.load_state_dict(state["state_dict"], strict=True)
        actual = model.state_dict()
        assert set(actual) == set(state["state_dict"])
        assert all(actual[k].shape == v.shape and torch.equal(actual[k].cpu(), v.cpu())
                   for k,v in state["state_dict"].items()), "Loaded tensors differ, including possible alias conflict"
        embedded = state.get("hyper_parameters", {}).get("codebooks")
        sid = torch.load(spec["sid"]["path"], map_location="cpu", weights_only=True)
        if isinstance(embedded, torch.Tensor):
            assert torch.equal(embedded.cpu(), sid.cpu()), "Checkpoint embedded SID differs from specified SID"
        strict = {"missing_keys": list(loaded.missing_keys), "unexpected_keys": list(loaded.unexpected_keys),
                  "global_step": state.get("global_step"), "epoch": state.get("epoch"),
                  "all_parameter_keys_shapes_values_equal": True, "state_tensor_count": len(actual),
                  "embedded_sid_exact": torch.equal(embedded.cpu(), sid.cpu()) if isinstance(embedded, torch.Tensor) else None}
        write_json(output / "checkpoint_protocol.json", {"protocol": "weights_only=True; explicit checkpoint-local safe_globals",
            "static_pickle": __import__("experiments.beauty_extensions.checkpoint_info", fromlist=["globals_in_checkpoint"]).globals_in_checkpoint(spec["checkpoint"]["path"])})
    dm = hydra.utils.instantiate(cfg.data_loading.datamodule)
    OmegaConf.save(cfg, output / "effective_config.yaml", resolve=False)
    used = {key: OmegaConf.to_container(cfg[key], resolve=True) for key in ("model", "trainer", "optim", "eval")}
    used["datamodule"] = OmegaConf.to_container(cfg.data_loading.datamodule, resolve=True)
    write_json(output / "used_config.json", used)
    write_json(output / "loading.json", {"assets": verified, "strict_load": strict,
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "shared_sid_table": model.get_embedding_table("encoder") is model.get_embedding_table("decoder")})
    return cfg, model, dm


def batch_keys(batch):
    opaque = batch[0].user_id_list
    if isinstance(opaque, list) and opaque and all(isinstance(v, str) for v in opaque):
        return opaque
    values = batch[0].transformed_sequences.get("user_id")
    if values is None:
        raise ValueError("Real user keys are required")
    return [hashlib.sha256(("beauty-extensions-v1:" + str(int(v[0]))).encode()).hexdigest()[:24]
            for v in values]


def batch_to(batch, device):
    import dataclasses
    import torch
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        if dataclasses.is_dataclass(value):
            return type(value)(**{f.name: move(getattr(value, f.name)) for f in dataclasses.fields(value)})
        if isinstance(value, (list, tuple)):
            return type(value)(move(v) for v in value)
        return value
    return move(batch)


def make_loader(cfg, split, limit_rows=None):
    import hydra
    from functools import partial
    from torch.utils.data import DataLoader, IterableDataset
    from src.utils.file_utils import list_files
    from src.data.loading.utils import assign_files_to_workers
    lc = hydra.utils.instantiate(cfg.data_loading[f"{split}_dataloader_config"].dataloader)
    dataset = lc.dataset_class(dataset_config=lc.dataset_config, data_folder=lc.data_folder,
        should_shuffle_rows=lc.should_shuffle_rows, batch_size=lc.batch_size_per_device,
        is_for_training=split == "train", assign_all_files_per_worker=getattr(lc, "assign_all_files_per_worker", False))
    files = list_files(lc.data_folder, suffix="*" + lc.dataset_config.data_iterator.get_file_suffix())
    assigned, _ = assign_files_to_workers(files, total_workers=1, assign_by_size=lc.assign_files_by_size,
        should_shuffle_rows=lc.should_shuffle_rows, assign_all_files_per_worker=getattr(lc,"assign_all_files_per_worker",False))
    files = assigned[0]
    dataset.set_list_of_files(files)
    dataset.set_distributed_params(total_workers=1, global_worker_id=0)
    collate = partial(lc.collate_fn, labels=lc.labels, sequence_length=lc.sequence_length,
        masking_token=lc.masking_token, padding_token=lc.padding_token, oov_token=lc.get("oov_token", None))
    if limit_rows is not None:
        from itertools import islice
        source = dataset
        class BoundedRows(IterableDataset):
            def __iter__(self):
                yield from islice(source, limit_rows)
        dataset = BoundedRows()
    return DataLoader(dataset, batch_size=lc.batch_size_per_device, collate_fn=collate,
                      num_workers=0, drop_last=lc.drop_last)
