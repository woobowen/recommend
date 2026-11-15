# Beauty collator repair: partial evidence, not a preflight RUN

Status: **PARTIAL / BLOCKED**. This is not a BEAUTY_PREFLIGHT_RUN, formal
training RUN, or READY_FOR_FORMAL. The historical failure remains unchanged in
`BLOCKED_BOUNDARY_PREFLIGHT.md` and `evidence/boundary-model-r0/`.

## Change and CPU result

`AuditTrainer.get_eval_dataloader` temporarily passes the original LETTER
collator to the inherited Trainer loader constructor and restores the indexed
training collator in `finally`. Neither dataset, training loss, validation loss
nor sampler was altered. `collator_real_preflight.py` extracts that exact
method from the production source and runs it with installed Trainer 4.31 on
real Beauty train/valid samples, without a model forward. It checks first,
last, short/long history, batched tensor identity with original collator,
labels, masks and target strings. The raw stdout/stderr are in
`evidence/collator-cpu-r0/`. This single-process CPU check is not distributed
validation, actual model training, or checkpoint acceptance. Production and
probe script SHA-256 are respectively
`77e0a2fa17402ddb177d974365bbd0cd20b54e3a84d544bf9fc36c69e5ab1f61` and
`7452f2bd8acfcd7c69adca3d03b5f7a311235d86012142e3a2cf251eb3eb4c56`.
Server test files at `/home/bwwu/grid-audits/sid-extensions-p0/transfers/beauty-collator-preflight-20261115/`
matched the corresponding local file hashes. Existing environment was used;
no packages or configurations were installed or changed.

## Unresolved execution requirements

No new GPU job was launched. No Continuous, Early or Resumed trajectory exists
for this repair; four validations, partial checkpoint, strict cross-process
comparison, best adapter, Normal and Shuffle remain unverified. The inherited
boundary guard undercounts the already consumed budget: it hardcodes 117.2274
instead of including the failed model attempt, and it runs Normal only. It
must not be reused as-is. Normal's current preflight chooses the first 8 users
per rank, while Shuffle acceptance requires a separately selected 16 users per
rank and reads their Normal records. The assignments and reference directory
must be made consistent, then the end-to-end interfaces need CPU checks before
GPU launch. No standards or methods have been relaxed.

Server job receipts queried on this turn: three toy jobs exited 0 and charged
28.825041640549898, 30.232078190892935, 58.17027708142996 GPU-device-seconds;
the earlier actual-model job failed with exit 1 and charged 197.967664770782.
Conservative total is 315.1950616836558; remaining of 3600 is
3284.804938316344 GPU-device-seconds. All four report no live worker. At the
resource query, two 4090s were each at 1 MiB with no compute processes; this
is a point-in-time observation, not a reservation.

Two non-author agents were asked to inspect training/recovery and generation
independently. Both reported they could not access files or execute tools. No
independent technical PASS exists. Successful full delivery requires real
access to the fixed source and raw receipts, the authorized full chain, and
independent audit; this report is not a substitute.
