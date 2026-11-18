# G3 conversion and CPU checks — PARTIAL

This is not a runnable G3 training or evaluation release. The existing production
entry points still bind the old LETTER index, vocabulary, checkpoints and catalog.
No formal training or G3 GPU preflight was launched. Direct non-author review was
unavailable: three separately started agents reported that they had no filesystem
or execution tools. No independent PASS is claimed.

The fixed upstream audit and its limitations are in `../identity_audit_abfe/`.
The new index is generated from the pinned tensor, complete bridge and LETTER
item axis by `convert.py`. Two executions produced the identical output; existing
different outputs are rejected. The four-digit token mapping is `<a_N>`, `<b_N>`,
`<c_N>`, `<d_N>`, preserving each original integer. `conversion.json` records
the resulting identity and counts. `negative_conversion_check.py` deliberately
overrides input hashes only in its synthetic mutation tests to reach the structural
guards; this is not a production mode. See `negative_result.json`.

The real LETTER Dataset and base tokenizer were exercised by
`cpu_dataset_check.py`: 131413/44726/22363 train/validation/test examples,
786 new tokens, vocabulary 32786 and allowed sets 256/256/256/18 plus EOS.
The tokenizer save/reload mapping and one two-example collator batch passed.
`cpu_shuffle_check.py` exercised the existing shuffle transform on all 22363
real test inputs and checked sampler partition identity. It is CPU input
evidence only, not model generation or an exhaustive negative test of boundaries.
Raw output is in the accompanying `.log` and `.json` files.

Execution used `/data/bwwu/envs/letter-20261110/bin/python`, CPU only, with
`OMP_NUM_THREADS=2` and empty `CUDA_VISIBLE_DEVICES`. The saved scripts and
results are at `/home/bwwu/grid-audits/sid-extensions-p0/g3-lcrec-prep-20261118-r0/`.
The fixed G3 and LETTER input paths are specified by the task and the audit;
the source path and base model are recorded in `cpu_dataset_result.json`.
The CPU commands were synchronous SSH invocations, not independent runner jobs;
the four runs returned exit code 0. No GPU device time was consumed.

Remaining work includes the full production identity and downstream adaptations,
the remaining CPU negative tests, evaluator and formal-entry dry-run, direct
non-author pre-GPU review, actual dual-GPU lifecycle and generation checks,
and direct post-run review. This partial evidence must not be used to approve a
formal launch or represented as completing those checks.
