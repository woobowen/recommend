# A1 Qwen training/checkpoint: nonauthor static review

Reviewer: `/root/qwen_prelaunch_review`; not a training/checkpoint/comparator implementation author. No SSH or GPU execution was performed by this reviewer. The runtime sources were bridged by the main controller and checked locally.

Frozen source commit: `a9d36910e20b95ab7df6622f29479b5d542f0b68`. All three reviewed core Python files and both configs were reread from this exact Git object and matched the reviewed working bytes. This report does not claim the later server deployment bytes or CPU job outcome were independently verified.


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

## Round1 targeted optional-backend correction review

Decision remains **STATIC_PASS_TO_BOUNDED_GPU_VALIDATION**, with the real failed round0 preserved. This section does not claim GPU, training, checkpoint restoration or A1 PRELAUNCH PASS.

The reviewer directly read the retained request, runner status/cleanup, pipeline status, both rank failure JSONs and empty rank event files under `evidence/jobs/letter-a1-q06-20261113-continuous-r0/`. Both ranks fail identically at `train.py:72`, importing `Trainer`: Transformers integrations import `TFPreTrainedModel`, which imports `activations_tf`; `tf_keras` is absent and installed Keras3 triggers the compatibility ValueError. This is before the entry's `torch.cuda.set_device`, distributed initialization, dataset loading and model construction. Both event files are zero bytes. The control flow and exact traceback establish zero completed optimizer updates for this attempt; no partial training state is claimed.

Retained runner status is `FAILED`, exit1, cleanup verified, `worker_may_be_live=false`. The pipeline records wall `10.431681293994188` seconds and charged GPU-device-time `20.863362587988377` seconds despite the zero-update import failure. The reviewer inspected that recorded charge and unchanged common preflight-ledger path/code; a new independent live-ledger query was not performed by this reviewer.

Raw failure evidence SHA-256:

| Relative path below the retained job | SHA-256 |
|---|---|
| `job/status.json` | `d30e56144f6599e5fb4d4b8e2e9a66254a80f61a6bfa28aea75530ffb0df77fd` |
| `job/request.json` | `4ebf42841543026b790f93507e412d280e1e2081e2c7ef58a0899c68548f954a` |
| `job/work.log` | `a321d19e3ea98c043ac683ccfee6739e77aefe58b3205f1a083d43badf8678c9` |
| `run/training/failure.rank0.json` | `5af539da4cad71ab6303f8ef9f504b061fc3c6f3188c317521d3322ae267d2c6` |
| `run/training/failure.rank1.json` | `5af539da4cad71ab6303f8ef9f504b061fc3c6f3188c317521d3322ae267d2c6` |

The round1 correction sets request-local `USE_TF=0`, `USE_FLAX=0` and removes `FORCE_TF_AVAILABLE`. No package installation or training method change is part of this correction. The existing experiment is PyTorch-only, and the historical Q06-L request already disabled TF/Flax, so selecting the intended backend restores the historical engineering environment rather than changing Qwen computation. Actual round0 request did not contain these two disabling variables.

The reviewer reread both actual CPU import receipts:

- `evidence/round1_cpu_trainer_import.json`, SHA-256 `259b768feed8667401be3f4f9b4119bff108eb0210fa4faa30392b13c557fc2e`: native Trainer import passes in the pinned Qwen interpreter, optional-backend flags are0, CUDA remains uninitialized.
- `evidence/round1_cpu_full_stack_import.json`, SHA-256 `a7c686e4db1997c8d18ea5e87bab0a130a559ffb2210555e85bba6bebabfa06f`: Trainer, callback, arguments, Qwen3 class, PEFT, Accelerator, native loader/sampler, DeepSpeed and bnb imports pass; Torch available, TF/Flax unavailable, no CUDA context. These are main-controller-bridged server CPU receipts, not an independent reviewer execution or GPU test.

All three core training/checkpoint/comparison files and both JSON configs were reread against fixed commit `a9d36910e20b95ab7df6622f29479b5d542f0b68` and remain byte-identical. Current control/pipeline diffs additionally allow explicit continuous/resumed comparison paths bounded under the A1 root and add a lightweight observation reader; they do not change the comparison standard, selected data, optimization, model, attention or generation. Round0 failure is not deleted, and the correction does not reset the task-level three-round limit or GPU budget. Actual round1 request/deployment and GPU outcomes still require their own evidence.

## Round2 targeted native metadata correction review

Decision: **STATIC_PASS_SCOPED_METADATA_CORRECTION**; no GPU or A1 PRELAUNCH PASS. Task correction rounds consumed: **2 of 3**. Actual installed-stack metadata-gate outcome and the next GPU outcome remain pending in this section.

The retained `continuous-r1` attempt uses RUN `851425db7de3c70c6c4925740ecc20d38bae3686`. The reviewer read both rank failure tracebacks, event logs, runner terminal/cleanup and pipeline status under `evidence/jobs/letter-a1-q06-20261113-continuous-r1/`. The failing call is the report JSON write in `on_train_begin`, reporting `Out of range float values are not JSON compliant: inf`. Both ranks' final event is `probe_initial_state`, with HF and DS global steps0 and actual BF16 Parameters; neither contains a batch or optimizer-step event. Thus this attempt loaded/initialized the model but completed zero optimizer updates. Runner status remains FAILED/exit1, cleanup verified and `worker_may_be_live=false`; pipeline charged `39.43586024083197` GPU-device-seconds for `19.717930120415986` seconds wall. Original failure is preserved.

Key retained evidence hashes:

- r1 `job/status.json`: `c735f1a66111f348cd5f022deb00368e5546c56cfd82bcba3ee605c218892a91`.
- r1 `job/work.log`: `3dad1f5629d8af9e9f846baca78cbd15247ac8f785cf336e341aaa6491c73d10`.
- r1 `run/training/failure.rank0.json`: `ba1ec68f5709c49a31487a4f206e828fb2197e5a8714d057662c3a6909ffbcb6`.
- r1 `run/training/events.rank0.jsonl`: `bdfd1940ea8ceab07b1f82b1a8344490a6fa5273a506d65457e0d109a6e14210`.
- `evidence/round2_native_metadata_source.json`: `8b546046c35459b6c6125865d7fe89f0f653159470888d1eba9ae91936ca971e`.

The bridged actual installed Accelerate source receipt identifies `utils/dataclasses.py` SHA-256 `6f6db1018c1be71436e34baec4b48ef3add91be8056a884cb032576dd867e90d`, line1290: `DeepSpeedPlugin.__post_init__` writes `steps_per_print=float("inf")` to suppress stdout logging. This is a native logging-control sentinel, not training loss or an optimizer value.

The author added `deepspeed_metadata` to the checkpoint helper. It copies the top-level DS config dictionary and encodes only a built-in float positive infinity at the exact top-level `steps_per_print` key as an explicit provenance-bearing string. It then validates the entire copied subtree with `json.dumps(..., allow_nan=False)`. It does not sanitize arbitrary floats, mutate the original config, change the DeepSpeed engine, or modify global writer/event strict JSON policy. The two production callers are limited to `report.training_arguments.deepspeed` and `report.effective_ds`. All loss/gradient finite assertions and the compute/checkpoint/replay/comparison logic remain unchanged in the reviewed diff against r1.

The reviewer independently imported the actual production helper locally, with no Torch or CUDA, and verified positive-sentinel encoding, finite2000 unchanged and original dictionary unchanged. Six negative cases were rejected: NaN at the allowed key; negative infinity there; positive infinity at an unrelated key; positive infinity at a nested same-name key; nested NaN; infinity inside a list. Global `write` and event functions still explicitly use `allow_nan=False`.

Reviewed round2 working bytes:

| File | SHA-256 |
|---|---|
| `scripts/train.py` | `b145d2f5606a3fa0703d39984860ade9401820d9f9cd2a750f8c21e499dcb88a` |
| `scripts/checkpoint.py` | `11e6191c44abd7402c1e0720e1358bfe0edbf86b28b4dc12f1717daacf79a1cc` |
| `scripts/metadata_check.py` | `fad418ae7cd2e18100b4948188098c3e259659c241f6106cc52cc124a41da9d7` |

`metadata_check.py` is a separate CPU-only installed-dependency gate. It exercises the real DeepSpeedPlugin and TrainingArguments aliases, asserts no original config mutation and no CUDA context, and verifies unknown nonfinite values remain rejected. Its `use_cpu=True` and disabled BF16 setting apply only to this metadata gate; those settings do not enter formal training. This is a report-serialization engineering correction within the authorized recipe, not permission to alter any training or restoration criterion. Fixed round2 deployment, actual CPU gate and real resumed numerical acceptance still need their own receipts.

### Actual installed-stack CPU gate received

The reviewer subsequently read `evidence/round2_metadata_contract.json`, SHA-256 `ba70f6a8985e4a66d59581c2bba8d029b9af43ff7abda4b61f7cd0f6e103d49b`, 4,824 bytes. It records `PASS_CPU_METADATA_CONTRACT`, fixed export `d9de77adfc92328f2bfd5e64ba40b652946f096f`, actual interpreter argv/cwd, empty CUDA visibility and TF/Flax disabled, plus `cuda_context_created=false`. Torch2.6.0 / Accelerate1.12.0 / Transformers4.57.6 / DeepSpeed0.16.9 match the reviewed environment. The config SHA is the previously reviewed `e4db1e7ed995a6733b754cbfed53af9f5191fb37a4be3eba33481d3a13a31fae`; actual production checkpoint helper SHA is `11e6191c44abd7402c1e0720e1358bfe0edbf86b28b4dc12f1717daacf79a1cc`.

The actual gate distinguishes two real TrainingArguments cases: ordinary requested config retains finite2000 because HfDeepSpeedConfig copies its input, while effective plugin config has the native positive-infinity sentinel; an already-resolved plugin config supplied to another TrainingArguments object carries the inherited sentinel. Both metadata encodings preserve the actual original configuration, and all five installed-gate unknown-nonfinite cases plus NaN loss / infinite grad-norm observations are rejected. This corrects the preliminary gate's aliasing assumption without changing production helper behavior or the training recipe.

The reviewer reread the executed gate source and verified its SHA `869cb580cfd434c75169c7d80ac38cec77e60ca260ed1295fe865b6cb683c44b` against fixed export commit `d9de77adfc92328f2bfd5e64ba40b652946f096f`; production train/checkpoint bytes likewise match that commit and the prior reviewed hashes. The earlier table records the pre-execution gate version, while this paragraph identifies the actual executed version.

Decision for this correction: **PASS_CPU_METADATA_ENGINEERING_FIX_ONLY**. This closes only the specific installed-stack report-serialization defect. The receipt was executed and bridged by the main controller; the reviewer independently read its source and evidence but did not SSH or repeat the server execution. Real GPU training, fresh-process checkpoint continuation/numerical comparison, best/validation and Beam20 acceptance remain outstanding. No additional correction round is created by receiving this evidence: the task remains at2 of3 rounds.

## Native tied-module configuration: targeted round3 basis

The real `continuous-r2` attempt reached two-rank `PROBE_COMPLETE` at12 updates using RUN `d9de77adfc92328f2bfd5e64ba40b652946f096f`. The reviewer directly read both retained training reports/events: they include complete checkpoints2/12, a real32-record validation interface, native strict hot restoration of checkpoint12, and `preflight_adapter` export. These are bounded training/interface events, not four-epoch or final-test results. The surrounding attempt then failed at the inference audit's requirement that saved `modules_to_save` literally equal `{embed_tokens,lm_head}`. Fresh-process numerical acceptance and actual Beam20 remain pending here.

That literal assertion was erroneous for the pinned native PEFT implementation. The earlier static review did not identify this native config rewrite; this review corrects that omission using the original retained files and actual installed source. No trained model, adapter or sharing structure should be changed to satisfy the mistaken string assertion.

Current saved `preflight_adapter/adapter_config.json` SHA-256 is `6e48b18ddd5201a518e0956a59287894902767c47dd672110cf27f4ea7a6bcde`. Its `modules_to_save` is `["embed_tokens","model.embed_tokens"]` and `ensure_weight_tying=true`. After normalizing only the order of `target_modules`, the entire current config equals the retained real historical Q06-L adapter config, not merely this one field.

The reviewer first inspected the retained PEFT0.21.2 wheel, SHA-256 `106ab6077ff72c54d21577f9af5970e34bac7582cb14209f7b2b511e322a4eae`, and then checked the controller-bridged actual installed source files against that wheel byte-for-byte. All match. The installed-source receipt is `evidence/round3_native_tying_sources.json`:

| Source | SHA-256 |
|---|---|
| `peft/tuners/lora/model.py` | `321964321e14ca0d9b4d81c4debcb486bae02193a576755b116ce35252e33555` |
| `peft/tuners/tuners_utils.py` | `e7f99b2ac5766ea317bcb943619d8a2a2235057acb5b543a1973ae8938173659` |
| `peft/utils/other.py` | `6b87957915ba399bfb6bbd57b9404aa12f0e9f4e27b19ef1b6c68ac052148f9b` |

Actual native causal chain:

1. `tuners_utils.py:1417–1443` detects tied embedding modules and `ensure_weight_tying`, then calls the LoRA-specific tie handler.
2. `lora/model.py:1293–1326`, `_add_modules_to_save_to_tie`, puts the tied head names into `modules_to_tie`, appends the fully qualified input embedding name to `modules_to_save`, and removes head matches from the ordinary `modules_to_save` list. The observed list therefore has the expected native rewrite.
3. `utils/other.py:1554–1566` wraps the tied destination modules with `tied_module` pointing to the active input-embedding adapter.
4. `ModulesToSaveWrapper.update`, `utils/other.py:647–651`, explicitly assigns `new_linear.weight=tied_module.weight`; the head uses the same active Parameter, rather than an independently optimized clone. Its forward delegates to the active `modules_to_save` module (`614–617`).

Recommended strict audit replacement: retain the fixed PEFT version/source, `ensure_weight_tying=true` and all LoRA recipe checks; recognize this exact native normalized config form rather than rewriting the adapter. After native load, require `modules_to_tie` to identify the actual `lm_head`, both input/output ModulesToSaveWrapper active-default modules to resolve to the full `[152587,1024]` BF16 matrix, and identical Parameter/storage. Keep all392 LoRA A/B tensors plus one independent active vocabulary Parameter, both saved embedding/head aliases, exactly394 adapter keys, exact expected shapes/dtypes/values, and no extra/missing keys. Both saved aliases must agree. Those semantic checks must execute on the real loaded inference model; simply deleting the string assertion would not be sufficient.

This is evidence for correcting an audit/compatibility check without changing the trained research method. It does not authorize untied weights, changed optimizer membership, relaxed adapter-key checks or repeated training of the already completed continuous path. The last round3 correction must remain concentrated with any genuine fresh-process comparison findings and within the unchanged cumulative preflight budget.

### Corrected inference audit static recheck

The reviewer reread the corrected `evaluate.py`, SHA-256 `c08f6cceb982147009e6d602a6b693065830e33a5b24e7d40b1d79b7156981f1`. It now accepts the exact pinned native config rewrite and requires native runtime `modules_to_tie={lm_head}`, both active-default ModulesToSaveWrapper objects, the same full BF16 `[152587,1024]` Parameter/storage,392 LoRA Parameters /5,046,272 elements, both saved tied aliases, and the entire394-key tensor set with explicit per-projection dimensions.

An additional strict-dtype gap was found and corrected within the same concentrated round3 review: the preceding check asserted only loaded BF16 dtype while converting saved tensors before comparison. The corrected source asserts saved and loaded dtype both BF16 and compares values without conversion. This prevents a malformed higher-precision payload from being silently rounded into an accepted checkpoint. No trained payload is modified.

The reviewer independently evaluated only the static expected-key/shape declarations and compared all394 entries against the actual retained historical Q06-L adapter-save inventory; all names and shapes match. This uses genuine historical runtime tensor metadata to verify the new shape assumptions, not a generated model or a new inference run. Static decision: **PASS_STRICT_NATIVE_TYING_AUDIT_TO_REAL_INFERENCE**. Real corrected Beam20 execution is still pending.

## Actual fresh-process training-state runtime review

Decision: **PASS_BOUNDED_FULL_STATE_RESUME**. This closes the new formal training-state restoration gate for the tested12-step path only. Overall A1 PRELAUNCH remains pending corrected inference/resource acceptance and the main-controller aggregate gate.

The reviewer read original requests, terminal/cleanup receipts, both sides' retained rank reports and full event files, checkpoint-completion manifests, and the actual CPU comparator result. All three jobs use RUN `d9de77adfc92328f2bfd5e64ba40b652946f096f`; requests pin train SHA `b145d2f5606a3fa0703d39984860ade9401820d9f9cd2a750f8c21e499dcb88a`, checkpoint helper SHA `11e6191c44abd7402c1e0720e1358bfe0edbf86b28b4dc12f1717daacf79a1cc`, comparator SHA `2a58c611318f96b315fb260d2dae6eaa577616e17bda62e2513aed2617747170`.

- `letter-a1-q06-20261113-continuous-r2`: its training component reached12 updates and exported the adapter; the overall job remains FAILED because its later original inference assertion failed. This review does not relabel that whole job successful.
- `letter-a1-q06-20261113-resume-r2`: fresh process strictly loaded continuous checkpoint2, replayed8 native-loader yields, performed10 further nonzero-LR updates to12, validated32 real records and strictly restored checkpoint12. Runner is SUCCEEDED/exit0 with verified cleanup and no possibly live worker.
- `letter-a1-q06-20261113-compare-r2`: actual CPU comparison over both two-rank acceptance states is PASS, SUCCEEDED/exit0, verified cleanup and no possibly live worker.

Actual comparator output is `evidence/jobs/letter-a1-q06-20261113-compare-r2/run/comparison.json`, 2,016,473 bytes, SHA-256 `0b3402cded34494fbcf22f372540a0439619975dcab799029d74227c5d7c495e`. The reviewer recomputed counts from every detailed comparison row: each rank has1363 tensors, every one finite and exactly equal, zero differing elements and zero tolerance violations. The original fixed criterion remains atol1e-6 /rtol1e-3; the observed result did not require numerical tolerance at all.

Each side/rank schema includes393 trainable tensors /161,295,360 elements; an80,647,680-element FP32 master partition; equally complete FP32 exp_avg and exp_avg_sq; all196 quantized projection dynamic states; scheduler, Python/NumPy/Torch/CUDA/loader-generator RNG and loader state. Each rank's serialized continuous/resumed acceptance files also have identical full SHA-256:

| Rank | Bytes per file | Continuous and resumed SHA-256 |
|---|---:|---|
|0|2,376,445,492|`d33e6cf54bc607323c0c51b29b88b24095241b2724edbe19737c9f0e8839e2c7`|
|1|2,377,416,948|`774cf6c3fdcfff413ee014a5aad3f6175dd07aa1174a90e6d3dcdaa763436eac`|

Independently of the comparator's event-summary booleans, the reviewer reread all retained raw event lines and recomputed the step2→12 joins. On each rank all40 microbatch input/label hashes, shapes and loader yields match exactly; all40 returned micro-losses and10 optimizer grad norms match exactly; LR sequences for steps3–12 match exactly and are nonzero/finite. The raw event-file SHAs also independently match the comparator's recorded input SHAs. Both fresh processes show strict restoration at2, `probe_initial_state` at2 with BF16 parameters, and `loader_replayed(replayed_batches=8, checkpoint_step=2)` before resumed batches.

Manifest inspections found23 complete files per checkpoint, both-rank collective identity, `exclude_frozen_parameters=false`, no duplicate or unsafe member paths, fixed run/config identity and expected step. Continuous checkpoint2 total manifest bytes4,417,916,589 /manifest SHA `107921af2696c87e80f48823013d6589ae527b45d6450742310fb1f9a0e1970c`; continuous checkpoint12 bytes4,417,916,640 /SHA `c7cc9be23fb453c033ddc462e65c1104311c55fe3cfd913275bc052bbede81a9`; resumed checkpoint12 same byte total /manifest SHA `d6667e9a9184e1b850b7db241d37de11b6086252954f948a2b8b6bd6964bd48d`.

Independence boundary: the main controller performed server/GPU/CPU execution and bridged the raw receipts; this nonauthor reviewer checked the original source, deployment identities, detailed comparison rows, native state coverage, raw logs and manifests. The reviewer independently recomputed event joins locally, but did not personally SSH, rerun GPU work, or unpickle the multi-GB acceptance files locally (Torch is unavailable in the local review interpreter). No such independent execution is claimed.
