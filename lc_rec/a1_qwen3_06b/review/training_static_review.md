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
