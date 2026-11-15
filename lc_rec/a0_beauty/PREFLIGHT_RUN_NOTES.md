# Beauty bounded preflight: execution inputs and limits

This snapshot is for `BEAUTY_PREFLIGHT_RUN`, never the full Beauty experiment.
Four short epochs use 10 microbatches per rank per epoch, GAS 4, and 32 real
validation rows. The preflight must terminate before formal training. Selection
is frozen in `config/preflight_users_seed42.json` (SHA-256
`714450a0e9479ed272266a5b4d3c2150d83742e4b70c38f80da888d4fa0aaa39`),
16 users per native shard. Normal and Shuffle reference this same file.

`preflight_budget.py` reads terminal runner status and the individual guarded
charges from all `letter-a0-beauty-*` jobs, requires the four historical jobs,
rejects unresolved or ambiguous jobs and reserves 60 dual-GPU wall seconds;
new work is capped at 1500 wall seconds or the smaller remaining balance.
All seven stages, including CPU compare/bind, are conservatively charged as
two devices for the guard's entire wall interval. `preflight_subprocess.py`
propagates child failure and terminates on deadline. The outer verified runner
is responsible for owned-process cleanup. Guard completion alone is not PASS.

`evidence/users-cpu-r0` contains outcome-blind sample generation and a CPU
bind using the tokenizer saved by the failed old attempt; this validates an
interface only, not a trained adapter or real-model inference. CPU regression
checks budget receipts, duplicate/missing/invalid samples and subprocess exit
and timeout. Existing full-user Shuffle CPU audit is retained; no outcome-based
selection or new permutation semantics were introduced.

Historical three correction rounds plus the later authorized Collator round
remain unchanged. This snapshot contains the newly authorized concentrated
correction. Independent reviewers attempted in this turn reported no file or
execution access, so no nonauthor audit PASS exists. The bounded GPU run and
its numerical evidence, not this note, determine technical status.
