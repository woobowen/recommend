# Beauty: training accumulation boundary requiring a decision

This is an unresolved method/continuity question, not a preflight PASS or RUN.
No Beauty training has been started.

The pinned LETTER loader produced 131,413 training examples. In the installed
Accelerate 0.21.0 `BatchSamplerShard`, micro-batch 8 and world size 2 yield
8,214 micro-batches per rank per epoch (`trainer_length_probe.py` and its
recorded output). The pinned Transformers 4.31.0 Trainer computes
`num_update_steps_per_epoch = len_dataloader // gradient_accumulation_steps`
and `max_steps = ceil(num_train_epochs * num_update_steps_per_epoch)`. With
GAS 4 and four epochs this gives 8,212 optimizer updates. Its loop tracks
`total_batched_samples` across epochs, and stops once `max_steps` is reached.
Under a fresh run with no earlier stop and updates every four micro-batches,
this stops after 32,848 of the 32,856 micro-batches needed for four full
passes: the last epoch is short by eight micro-batches per rank. This is
derived from the installed source and an actual shard-length probe, not a
Beauty training observation.

Moreover, 8,214 is 2 modulo 4. Epoch-1 and epoch-3 validation and checkpoint
therefore occur between accumulation boundaries in that loop. The historical
A0 checkpoint mechanism was accepted for the even Instruments schedule; it
has not been shown to save and restore Beauty's partial accumulated gradient
and accumulation phase. DeepSpeed 0.16.9's engine model checkpoint lists
global steps but not `micro_steps`; this alone does not prove exactly how
its optimizer files treat all partial gradients. A continuous/resume
comparison at such a boundary is required before claiming strict recovery.

The original Trainer behavior, a complete four-epoch traversal, and strict
resume equivalence cannot be assumed equivalent on this dataset. A full-pass
implementation would require 8,214 updates if all 32,856 micro-batches are
consumed in groups of four; it also requires a verified treatment of
intermediate validation/checkpoint accumulation state. Merely increasing
`max_steps` is insufficient evidence of correct recovery or method equivalence.
The user has approved a control and recovery adaptation to complete four full
epochs with continuous gradient accumulation, subject to strict validation,
but also expressly prohibits repeating samples to obtain the step count.

The original length probe overlooked a material fact: Accelerate 0.21's
`BatchSamplerShard(even_batches=True)` completes the final pair of batches by
repeating 11 examples per epoch. `training_sampler_padding_probe.py` uses the
installed class and a fixed synthetic permutation to verify 131,424 slots for
131,413 distinct examples. Four passes without repeats contain 525,652 example
occurrences = 8,213 full batches of 64 plus 20 examples; 8,214 full batches of
64 require 525,696 occurrences, 44 more. The 8,214 full-update claim therefore
depends on the original sampler's 44 repeated slots across four epochs. This
was already the native sampler policy, not a new padding rule invented here,
and the user has explicitly approved retaining this native padding as a
limited exception. The formal GPU chain remains gated on the required
cross-epoch accumulation and strict recovery acceptance.

Read-only installed-source identities on `zwlab-01`:
`accelerate/data_loader.py` SHA-256
`7741d477a6fb02f18f8e5871d348b963fc669a51821f43102215255f02c48cb8`;
`transformers/trainer.py` SHA-256
`24a579e7af77c3699020ac16c1b4ad16d64aa584dafe52a90d0807c381174f31`;
`deepspeed/runtime/engine.py` SHA-256
`88db49f4970ad38101946467c69abc07b67f4454d4411eea1dbc13ed9c2f460d`;
`deepspeed/runtime/zero/stage_1_and_2.py` SHA-256
`7e03bb5b2c923b2291ed28b663b1b605aa1384f68aa23b5744951183cc42fef6`.

`accumulation_boundary_model.py` tests a small synthetic CPU state machine at
both partial and complete accumulation boundaries, comparing pending gradients,
parameter value, step counts, RNG, input order and validation positions after
serialization. It is not a test of Hugging Face Trainer, DeepSpeed, GPUs or
the real checkpoint implementation.

`accelerate_epoch_probe.py` exercises the actual installed Accelerate on short
CPU epochs. Its saved `*_1.stdout` demonstrates the default end-of-dataloader
accumulation reset; `*_0.stdout` demonstrates continuous synchronization
positions when that reset is disabled. Neither run establishes DeepSpeed
optimizer behavior or validates an actual Beauty run.

## Existing preliminary evidence

- The actual dataset loader and target split were checked for all rows;
  response-only mask was checked for four examples (`loader_audit_r2.stdout`).
- The fixed 965-token vocabulary was reconstructed and single-token mappings
  verified. No Beauty model or adapter exists yet.
- `uneven_sampler.py` passed even-dataset equivalence, actual Beauty 11,182 /
  11,181 allocation, DataLoader ordering, and missing/duplicate rejection.
- `shuffle_cpu_preflight.stdout` covers all 22,363 test prompts with a
  reconstructed tokenizer and the fixed historical core. It is not the
  required saved-adapter or model-level preflight.
- The first loader invocation lacked the historical CUDA environment; a
  first repair kept GPUs hidden and hit DeepSpeed's CUDA capability check.
  The second environment repair used the historical CUDA visibility and
  succeeded. No model training occurred. These are the first two scoped
  engineering corrections of the current task; failure output for the second
  attempt is retained in `loader_audit_r1.stderr`.
- Independent agent `/root/train_adaptation` analyzed the Trainer arithmetic
  via a main-agent text bridge. It had no terminal or SSH and did not read
  source or execute the probe itself; it independently derived 2,053 / 2,054 /
  2,053 / 2,052 updates and eight missing last-epoch micro-batches. This is
  not the required direct-source, non-author technical preacceptance.
