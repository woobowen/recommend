# Beauty A0 formal launch and early observation

This is launch evidence, not final experimental results. The authorized experimental series is `beauty-a0-seed42-original`; task ID is `beauty-a0-seed42-formal`. The actual calculation RUN is `6e1b211d9c37f358c43b1540b40b17f6703150d2`; sealed Plan delivery is `002022ea051a4c4f490dd9dee46b87167f44304e`. Evidence delivery is a separate commit.

The user directly authorized the scope in the launch request. `authorization.json` is a ChatGPT-prepared transcription of a preceding user message, not a digital signature or a claim that the user manually wrote JSON. Its SHA-256 is `ca733239d9c95a35a4c5504f8dc205c1039d91399df29630ef0b48d5caaa1963`.

The sealed Plan raw SHA-256 is `f5a3e7cb81269f637b92d29e021462aa72a3dd3ee86ff1ec8dfb95e93e1a08fc`, canonical digest `0597958bcf852be1537252507a6c20e153256d3135f990b7362e45e867f95206`, and effective config SHA-256 `24b568161f2d71d3c5962372a38683b2d33ee5d6d5064de0612215946a8106fd`. The external review SHA is `3a18da80530451d72efde8aed46e7d51ad44b5bcd4b3396e579fab328de53a97`; the approved budget record SHA is `2bcd27d8b7c6dbdfe373e3fe97d9bbcc8ca9b2626e54d482241fea46f7936deb`. Reference the sealed Plan evidence at the delivery commit.

`launch.sh` is the actual saved script; `launch.log` captures its prelaunch checks, argv and start receipt. The server source is `formal-candidate-5b2917d/launch-evidence-20261116/launch.log`. `request.json` and `accepted.json` are copied from the original server request and Runner job. `observation-*.txt` are individually timed, non-atomic live observations, including bounded log tails; they do not represent an immutable filesystem snapshot. `observe.sh` is the read-only collection script. Hashes and final status are in `MANIFEST.sha256` and the separate final observation note.

The formal job ID is `letter-a0-beauty-formal-beauty-a0-seed42-formal-train-a0`. The plan root, job files and controller log paths are visible in the captured receipts. No result is inferred from startup alone.

Sealed Plan and external review: https://github.com/woobowen/recommend/tree/002022ea051a4c4f490dd9dee46b87167f44304e/lc_rec/a0_beauty/evidence/plan-seal-20261116
