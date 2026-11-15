# Beauty inference-only preflight RUN

The user explicitly authorized up to 3600 additional GPU-device-seconds for
bounded inference preflight only; historical 3600 and 3316.546833369881
charged remain unchanged. `inference_preflight_guard.py` reads the live job
receipts, requires the five historical identities, caps the project at 7200,
caps the new authorization at 3600, and reserves 60 dual-GPU wall seconds.
The fixed single job is `letter-a0-beauty-infer-20261116-r0`. Its only GPU
stages are 16-user/rank Normal and Shuffle acceptance on the same 32 users.
It cannot invoke training or the full test set. Guard terminal success means
stages ran, not independent acceptance or authorization for a formal job.

Input is the saved continuous preflight Adapter from training RUN
`f54a2b505fd2d8767109a11d330a7064aff8af5e`, SHA-256
`e56abd90bd3cc55d3786df04971a670bc12c54578110d255b5e7f3f4ec76af07`.
CPU binding and derived source checks in `evidence/inference-cpu-r0/` used the
actual saved tokenizer and fixed user manifest; CPU weight check verified 450
saved keys, shapes and finiteness, not a real PEFT model reload. Normal's
32-user output is the Shuffle reference. Formal training remains unauthorized.
