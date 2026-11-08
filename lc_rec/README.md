# LC-Rec Instruments / original LLaMA reproduction preparation

This is a one-time, bounded preparation task authorized on 2026-11-08. It does not authorize formal4epoch training, five-seed training, Qwen, another dataset, Shuffle, MLP, attention analysis, RL, CoT, or a new loss. Stop after publication and a validated lightweight review ZIP.

Official source: [RUCAIBox/LC-Rec](https://github.com/RUCAIBox/LC-Rec/tree/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669), commit `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`; [paper v4](https://arxiv.org/html/2311.09049v4). `vendor/original/` is an unmodified, Git-blob-verified source snapshot, not a submodule. No LICENSE file exists at this upstream commit; upstream provenance and notices remain intact. Model licensing is separate, preserved in `evidence/assets/hf_LICENSE` and the model card; public access is not a new license grant.

`config/task.json` records authorized preparation paths and limits. `config/formal_five_seed_candidate.json` records the complete future recipe and unresolved conditions. `scripts/render_candidate.py` **only prints** commands; it never executes or queues training. `config/bounded_prerun_candidate.json` records the proposed10update/32validation/16user limits; its GPU instrumentation is explicitly unimplemented/unverified while the hardware gate is blocked.

Use the six original tasks with Instruments sampling `1,1,1,1,1,1` / `0,0,0,20000,0,0`, response-only labels,20 historical items,2048tokens, full-parameter BF16 ZeRO3 without offload, global batch128, AdamW5e-5/0.01, cosine/warmup0.01 and4epochs for the future formal candidate. The two-device layout micro1×accum64 is an allowed layout candidate, not a memory solution or bitwise-identical trajectory.

Current resource gate: both real devices have24564MiB. Even8bytes/parameter for sharded FP32 Adam moments alone uses25.10GiB/device before weights, gradients, activations and communication. Consequently no GPU training optimizer/model-loading attempt or model weight download is justified in this task. CPU data/tokenizer checks must not be reported as forward/backward, parameter updates, complete model reload, generation, formal experiment completion, or paper reproduction.

Important original-source conflicts: `run.sh` has an erroneous space in `BASE_MODEL= huggyllama/llama-7b`; `finetune.py` loads with device_map before DeepSpeed setup; SID generation constraints are positional token sets, not a canonical prefix trie; `test_ddp.py` reduces beam after OOM; raw auxiliary generation uses an obsolete API and has a train/test dictionary condition issue. Original files remain unchanged. Do not silently patch research semantics, reconstruct labels, reduce beams, or claim exact-catalogue generation correctness.

Preparation uses an independent user venv while reading existing Python/torch components. Its pinned small dependency overlay is for CPU audit only, not the author's recovered historical environment. No global framework or GRID environment is changed. Exact imports, installs, assets, job state, hashes and audit results are recorded under `evidence/` at delivery.

`rules/` contains frozen audit copies and unified diffs of the three authoritative files in `/home/addaswsw/recommend`. Only conflicting branch layout clauses were updated: LC-Rec=`wbw-lc-rec`, old GRID lines=`wbw-grid`,`wmh-grid`,`yjc-grid`. These are not new maintained rule entrypoints. Web project copies have not been changed.

Delivery results and blockers are reported in `REVIEW_ENTRY.md` once actual checks finish. No status in this preparation commit anticipates a successful runtime check.
