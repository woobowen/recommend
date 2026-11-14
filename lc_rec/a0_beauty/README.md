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
Training, tokenizer, checkpoint, evaluation and Shuffle adaptations and their CPU/GPU
acceptance, independent reviews, and full experiment remain outstanding.
