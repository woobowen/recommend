# A0 Beauty: preliminary author data audit

Status: PARTIAL. This directory is not a training or evaluation RUN release.
No Beauty GPU job has been started and no experiment results are represented here.

`audit_author_data.py` downloads three files from the pinned LETTER commit, verifies
their Git blob identities and reports byte hashes and structural counts. Run it with
Python 3 and network access. `author_data_audit.json` records one successful execution.
The train and validation sample counts are derived from sequence lengths; they have
not yet been verified by running the actual SeqRec loader. Matching keys does not
independently establish item metadata semantics or equality to GRID records.

The historical A0 test sampler uses `drop_last=True` on two ranks. Beauty has 22,363
test users, so this sampler drops one user; `drop_last=False` pads a duplicate. A
validated, non-overlapping full-coverage assignment is required before formal test.
The local Normal evaluation and Shuffle entry adaptations are drafts; only
synthetic aggregation, sampler and preliminary full-user CPU checks have run.
The actual loader and tokenizer have now been checked; the response mask was
sampled. See `TRAINING_BOUNDARY.md` for the unresolved full-epoch and resume
semantics discovered from the actual pinned training software. Training,
checkpoint and evaluation/Shuffle adaptations, GPU acceptance, independent
final reviews, and the full experiment remain outstanding.
