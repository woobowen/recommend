# Qwen A1 historical basis: nonauthor verification

Reviewer: `/root/qwen_prelaunch_review`. Scope: local read-only inspection of fixed historical Git objects and retained original receipts; no SSH, GPU execution, model implementation, training implementation, or Git mutation. This reviewer wrote only this review document. The first shell command was `ls -la` in `/home/addaswsw/recommend`; it confirmed the three colocated project rule files. Those files were explicitly read from that external rule entry. No ancestor AGENTS/override was found for the LC-Rec worktree; `lc_rec/rules/AGENTS.md` is outside the reviewed scripts' ancestor scope.

## Historical evidence identity

- Historical delivery: `55a1371da28a45a2ec81ceee7bd82fd42b7192d5`.
- Actual Q06-L GPU RUN: `f29527de3b3f94ae5640d2e98faa1a7f74d2e0e8`.
- Actual historical job: `qwen-20261110-q06-l-r0`; retained status is `SUCCEEDED`, exit code 0, cleanup verified. These are historical records, not a current server query.
- Reviewed fixed files: `lc_rec/qwen_preflight_20261110/REVIEW_ENTRY.md`; `scripts/common.py`, `cpu_audit.py`, `probe.py`; `config/experiment.json`; `evidence/execution_results.json`, `summary.json`; model manifest; the job request/status/cleanup and both original rank records; CPU tokenizer audit; existing asset, runtime and independent cost reviews.
- Retained rank0 JSON SHA-256: `6271fc9c8839562a14c922e111b192eb4cc457e014a2e6a5ccf17d2c53bb6165`.
- Retained rank1 JSON SHA-256: `813288e49bdcd8843d394f6cbc352e9b8285464fc425ac9f71db1c563095a44c`.
- Both hashes were recomputed locally from fixed Git bytes and match `execution_results.json`.

Original rank paths within the fixed delivery:

```
lc_rec/qwen_preflight_20261110/evidence/jobs/qwen-20261110-q06-l-r0/output/probe/rank0.json
lc_rec/qwen_preflight_20261110/evidence/jobs/qwen-20261110-q06-l-r0/output/probe/rank1.json
```

The historical interpreter was `/data/bwwu/envs/qwen-preflight-20261110-qwen3/bin/python`; versions were Torch 2.6.0, Transformers 4.57.6, PEFT 0.21.2, Accelerate 1.12.0, bitsandbytes 0.45.5 and DeepSpeed 0.16.9.

## What the historical run establishes

- Official `Qwen/Qwen3-0.6B-Base`, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, class `transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM`. Manifest weight size 1,192,135,096 bytes and LFS SHA-256 `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` match the requested identity and retained complete official-asset verification chain.
- Original author Instruments A assets: index `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd`, interactions `0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743`, items `1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2`.
- Tokenizer 151,669 to 152,587; physical embedding/head rows 151,936 to 152,587. All 918 SID token identities are new to the tokenizer, but 267 use original padded rows and only 651 rows physically expand. `mean_resizing=False` and the original model initializer were used.
- CPU audit checked all 9,922 item SID encodings/decodings before and after tokenizer save/reload. The deterministic ordered `(SID token, ID)` list, encoded with JSON compact separators, has SHA-256 `87c3c3f1ed53136755427ddb66be46480ac428a38b38e2ba47ecd34190b5a5ee`.
- Original Base EOS and PAD are both 151643; no automatic BOS is inserted by this tokenizer. Padding labels use `attention_mask`, preserving real EOS supervision.
- 28 layers times seven projections gives 196 LoRA target modules and 392 A/B tensors. LoRA parameters: 5,046,272. Full active tied vocabulary matrix: 156,249,088. Independent trainable Parameters: 393; independent trainable elements: 161,295,360. The native adapter state has 394 keys because embedding/head have shared aliases.
- Same active embedding/head Parameter and storage were evidenced before/after resize, after PEFT, after DeepSpeed, and after adapter reload. Optimizer membership was unique and matched all trainable Parameters. Frozen original module copies did not enter the optimizer.
- Native Qwen SDPA, Int8 trunk and BF16 computation, ZeRO-2, nonreentrant gradient checkpointing, micro8 times accumulation4 times two ranks equals global64. No LLaMA FastChat, deterministic FA2 or A-only patch was used.
- Four genuine global updates, three with nonzero LR. All LoRA tensors had finite nonzero gradients and sampled changes; SID input/output paths had genuine gradients and active SID rows changed. FP32 full-vocabulary causal CE divided logits by 0.8 and matched an independent masked log-softmax formula within `9.5367431640625e-07`.
- Each sample supervised exactly four SID tokens plus real EOS. Original 17 templates, recent20 history and author splits remained. Dataset counts were 131,837 train rows, 49,544 validation texts, 24,772 test users.
- CPU full-length audit under the recorded template policy found train/validation/test maxima 184/184/167 tokens, with truncation unused. Training full-length audit used one fixed seed42 template draw per raw row, not every template combination.
- Strict native adapter save/reload checked all keys/values and shared weights. Historical inference explicitly used `autocast_adapter_dtype=False` and `get_peft_model_state_dict(..., save_embedding_layers=False)`.
- Eight genuine test users, 20 sequences per user, finite sequence scores and the position callback sizes 150/256/256/256/EOS1. All 160 sequences were position-legal but noncanonical. That is a bounded technical observation, not recommendation quality after training.

## What the historical run does not establish

Both original ranks explicitly record:

```
full_training_state_restore = "NOT TESTED"
full_optimizer_checkpoint_written = false
```

No formal four-epoch Trainer run, complete validation aggregation/selection, fresh-process full DeepSpeed optimizer/scheduler/RNG/loader restoration, best-checkpoint restoration, or all-user test was executed. Loader length was inspected, not a whole epoch traversed. The 8,240-update count and 83 warmup steps must be checked again in the actual formal entry.

The historical 18.0167156-hour center / 21.7596398-hour conservative seed estimates included explicit checkpoint I/O proxies and 10%/25% assumptions. They are conditional planning estimates, not confidence intervals or measurements of full-state restoration. Historical Q06-L did not record the whole post-training/pre-validation-reset allocator interval, so its saved training peak must not be presented as a precise full-job allocator peak.

## Historical code identity differences

Actual GPU RUN `common.py` SHA-256: `5185c6578e5716a163d33688505db62d5140fdaa6cc9512c280253a21d848c06`.

Actual GPU RUN `probe.py` SHA-256: `4dc56620b9427dc58b792d427cca9d0b4af160f9d8e3027e942026e408f9bd39`.

`cpu_audit.py` is byte-identical between the historical GPU RUN and delivery, SHA-256 `9cee4476071f4bbb9bdba95e375788929f90e741a889f8285ef2f4ed51f10d27`. Delivery `common.py` subsequently adds set/frozenset JSON serialization. Delivery `probe.py` subsequently adds budget reconciliation, allocator-environment metadata and post-training/validation peak accounting. The final probe is therefore not the exact historical Q06-L execution file, though these changes do not alter its earlier retained evidence.

## Reusable A0 checkpoint design and Qwen-specific boundaries

The following are byte-identical in the historical delivery and actual A0 GPU RUN `4c6a83d9ba42dc2fe737792759e410ef1874da76`:

- `lc_rec/letter/scripts/formal_train.py`, SHA-256 `7c18589c26e4a160ebe5bda14d47af0a05533820e0ba4c8c0a551fdbd74ee5a8`.
- `formal_bnb_reload.py`, SHA-256 `cb3d3e7a827950f9db30c81204ed66a2cb1fe70b7d538bfd237b212bc88b7ce2`.
- `formal_compare.py`, SHA-256 `b81679e499ff7c072264c2b60efd9ea0a0858b4d47a853c6ea0a6953e48a1bfd`.

Reusable designs include epoch-start/checkpoint RNG plus CPU loader replay (`formal_train.py:218–273`), all-rank complete markers before best/latest rotation (`343–372`), strict DS loading, and the comparator's recursive coverage of trainable tensors, FP32 masters, moments, scheduler, discrete state and all RNG streams. Keep `abs(candidate-reference) <= 1e-6 + 1e-3*abs(reference)` and exact required state; do not lower coverage to obtain PASS.

Do not copy A0 architecture/version assumptions: LlamaTokenizer, LETTER class, FastChat/FA2/A-only patches, 450 trainable tensors, 224 quantized projections, `tokenizer.model`, Accelerate0.21 `no_sync` patch, or HF4.31 scheduler ownership. Qwen uses HF4.57.6 and Accelerate1.12. The bnb hot-SCB design is only a possible compatibility mechanism; current cold/hot Qwen evidence and actual 196-module layout must determine applicability.

Modern Trainer can infer `model_accepts_loss_kwargs=True` from Qwen's `**kwargs` and use `num_items_in_batch`; formal code must verify per-micro CE and gradient accumulation normalization against the author semantics, including DeepSpeed's `scale_wrt_gas`. Another six-task Qwen project documents this engineering risk but its global128, different SID, LoRA alpha and loss settings are not A1 method authority.

## Prelaunch decision

Historical technical basis: **PASS, bounded technical scope only**. This document is **not** an A1 PRELAUNCH PASS.

Required new evidence before formal launch: fixed formal source/config and deployment bytes; current assets/environment/GPU/locks/ledger; 393 unique optimizer Parameters and tied storage; exact response masks and independent temperature CE; real two-rank complete checkpoint plus fresh-process nonzero-LR continuation; unchanged full comparison coverage/tolerance; exact inputs/labels/sampler/RNG; current-version best/latest atomicity and best restore; real validation interface; strict BF16 inference reload, longest-input Beam20 and resource measurements; preflight device-time at most 60 minutes and cumulative formal wall cap 96 hours. Review the actual new implementation and records before changing that decision.

## Round0 evaluation-entry static review

The reviewer did not implement `scripts/evaluate.py`. Its reviewed working bytes after author corrections have SHA-256 `83c502aefa47093efc7335970ca5767946bc551bad21afc7bee574146e7a3208`; these bytes were uncommitted and had not been deployed or run on GPU at review time. This is a static/CPU helper review, not a real inference PASS.

Two defects were found before execution and corrected by the implementation author:

1. Adapter loading omitted `autocast_adapter_dtype=False`. The current captured PEFT `peft_model.py:425` defaults this flag to True, its documented behavior at lines455 onward casts BF16/FP16 adapter weights to FP32, and lines1597–1600 call the casting path even before `is_trainable=False` switches to eval. That differed from the historical explicitly BF16 adapter inference. The author added the flag and hard assertions that every loaded adapter tensor is BF16; the reviewer reread both corrections.
2. `get_peft_model_state_dict(model)` omitted `save_embedding_layers=False`. The captured PEFT `utils/save_and_load.py:298–342` detects resized vocabulary and can append embedding-wrapper state outside the historical 394-key adapter schema. The author now requests False explicitly; the reviewer reread this correction. No key-coverage relaxation was used.

Captured dependency sources were read from `/home/addaswsw/recommend-support/a1-qwen3-06b-instruments-20261113/runtime_sources/`, with their source identity manifest. These are files bridged by the main controller from the server, not an independent SSH session. PEFT source SHA-256: `peft_model.py` `d22a1c50dea7a046ce28e71a79dfc20b1d96a2d238a9b457be6a88b6068e04d5`; `utils/save_and_load.py` `6afbb53933550a401a19e883ba888f4818eb507f64306f5619a56316b598289b`.

The corrected entry reconstructs the fixed sorted SID-token order and requires tokenizer vocabulary/added-vocabulary and EOS/PAD IDs to match; 394 adapter keys must match the loaded state, shared active Parameter/storage is checked, and all tensor values are checked after the explicitly stated dtype conversion. Real runtime evidence is still required for those assertions.

The per-rank inference batch is one. Its callback binds the full actual tokenized prompt, so no LLaMA separator slicing or left-padding offset assumption is imported. The bounded-user selection sorts all actual test prompts by their actual Qwen token length and includes the longest/near-longest entries on both ranks. The formal sampler assignment covers 12,386 distinct rows per rank, and aggregation requires every row exactly once, twenty candidates, finite scores, exact user/target mapping and stable ordering. This is source inspection, not measured all-user coverage.

Local CPU checks, without Torch/model loading or file mutation, passed AST parsing and exercised the actual vendor evaluator against `audit_prediction` with twenty synthetic candidates containing score ties, repeated target SID, an invalid SID set and a colliding target. Results were hit@5=1, hit@10=1, ndcg@5=`0.6309297535714574`, ndcg@10=`0.9871369406794797`; illegal candidates=17, duplicate occurrences=1, colliding candidate occurrences=2. A deliberately reversed saved ranking was rejected. These are synthetic regression values, not A1 metrics.

Provenance item remaining for the surrounding pipeline: evaluation aggregation currently compares adapter/data/source/generation identities between ranks, while the base-model weight SHA in its report is a pinned expected constant. The formal deployment/preflight/pipeline must supply actual current base-manifest/weight verification and tie the exported adapter to the chosen best checkpoint; a written SHA constant or revision-directory name alone is not actual asset verification. No final A1 PRELAUNCH decision is made here.

## Round0 training/checkpoint static review

Decision: **STATIC_PASS_TO_BOUNDED_GPU_VALIDATION**. The reviewer found no remaining static defect preventing the authorized bounded GPU acceptance. This is expressly **not A1 PRELAUNCH PASS**: no new GPU or real fresh-process restoration evidence was available at this review point.

Reviewed working-source SHA-256:

| File | SHA-256 |
|---|---|
| `scripts/train.py` | `44eee2cab4ac1da407418136e929bf257ecc71191c34348a6ac10c64ca6b8841` |
| `scripts/checkpoint.py` | `2a9cf3536125cf875ebd9d3806ef86d33883d8e50ec9c0c0f6cc89997518c717` |
| `scripts/compare.py` | `2a58c611318f96b315fb260d2dae6eaa577616e17bda62e2513aed2617747170` |

Both JSON configs and all current A1 Python entries parsed successfully. The DS buckets now retain the historical Q06-L 50,000,000-element settings; they are no longer the initially copied 500M values.

One additional observation defect was found before execution and corrected by the training author: the initial parameter-change snapshot was originally taken before DeepSpeed converted FP32 active weights to BF16 and before strict resume loaded checkpoint values. That could count dtype rounding or checkpoint loading as optimizer changes. The reviewer required moving the baseline into `on_train_begin`, after DS initialization and any strict restoration, before the first update of that process. The corrected source records initial HF/DS step and actual parameter dtypes. Gradient hooks remain observational and are absent in the formal unbounded path.

Core checks against captured actual installed sources:

- The custom loss delegates to the historical FP32 full-vocabulary `common.loss`, including division by temperature0.8. `model_accepts_loss_kwargs=False`, `compute_loss_func=None`, and no `num_items_in_batch` ensure the captured HF4.57.6 `training_step` divides the mean loss exactly once by `current_gradient_accumulation_steps=4`. That same native source passes `scale_wrt_gas=False` to DeepSpeed. Accelerate's actual wrapper sets accumulation boundaries and steps only on the synchronized microbatch. No historical Accelerate0.21 patch is applied.
- The captured native Trainer fetches four microbatches before each optimizer group, calls loader `set_epoch`, and attempts late RNG restore after fetching. The wrapper suppresses that late restore; ReplayLoader restores epoch-start input/sampler RNG, replays saved yields to reproduce the native DataLoaderShard one-batch lookahead, then restores checkpoint RNG before the next yield. `loader_generators` captures actual available generator objects and initializes the SeedableRandomSampler's lazy generator without consuming a random draw.
- Native model/adapter save explicitly uses `save_embedding_layers=False`, native bin serialization and full active tied matrix. Full DeepSpeed save overrides the modern HF PEFT default `exclude_frozen_parameters=True`, requesting False. Both native HF resume and native best restoration call the guarded hook enforcing `load_module_strict=True`, optimizer and scheduler restoration, complete-marker verification and step agreement. Native `scheduler.pt` and explicit external scheduler files are written.
- `strict_hot_int8_alias` only borrows existing SCB storage while native strict loading runs and restores its hot layout in `finally`. The corresponding real native CPU cold/hot serialization reproducer still must execute in the captured server environment; source inspection alone is not that result.
- Full-rank files, loader/RNG, adapter/tokenizer and Trainer state are required before sealing. Checkpoint rotation is suppressed during native writing and occurs after collective completion and seal verification. Best selection is native validation-loss selection; the same strict restoration path handles best reload.
- The bounded path keeps the full four-epoch scheduler/loader and stops at12 updates, saves at2 and12, and permits fresh-process restoration from2. The comparator requires 393 complete trainable Parameters / 161,295,360 elements, all196 Int8 projection states, FP32 master partitions with complete Adam moments, scheduler, all RNG streams and loader state. It compares the complete 40-microbatch interval after step2, exact input/label/order/LR, finite losses/grad norms and fixed tensor tolerance. No comparison-coverage or tolerance relaxation was found.

A separate local CPU mechanism check imported only the checkpoint module's actual ReplayLoader control flow and supplied explicitly synthetic Python input/sampler/compute RNG hooks plus a one-batch-lookahead loader. After checkpointing at2 updates, all16 remaining synthetic microbatches and final independent RNG streams matched on replay. This checks control flow only: Torch was unavailable locally, no model or CUDA was loaded, and the synthetic test is not real DataLoader/DeepSpeed or full checkpoint acceptance.

The current surrounding pipeline was also inspected for the earlier identity item: it consumes this round's CPU asset/environment hash receipt before GPU work, and verifies the final adapter SHA against `best_selection.json` before formal test. Actual CPU receipts, frozen deployment, bounded GPU continuation/numerical comparison, validation/best/inference resource evidence and real nonauthor runtime decision remain required.
