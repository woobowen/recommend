# Launch and observation outcome

At 2026-11-16 08:28:11 UTC, `start` returned controller PID 3280445. In a fresh connection the Runner accepted job `letter-a0-beauty-formal-beauty-a0-seed42-formal-train-a0`, with supervisor PID 3280453, worker PID 3280454, torch launcher PID 3280455 and ranks 3280460 and 3280461. Their command lines and process sessions are recorded in each observation. The raw Runner status records its start at 08:28:12.295791 UTC. The conservative observation window uses the first saved full snapshot at 08:28:45 UTC through 08:59:00 UTC: 1815 seconds. Controller and both ranks were present at both ends. No restart or retry was performed.

| Snapshot UTC | Latest recorded optimizer step / loss | GPU 0 / 1 memory | Runner |
|---|---|---|---|
| 08:28:45 | model initialization, no logged update | 9114 / 9114 MiB | RUNNING |
| 08:33:36 | 50 / 5.5377 | 16004 / 16004 MiB | RUNNING |
| 08:43:42 | 200 / 3.9966 | 16004 / 16004 MiB | RUNNING |
| 08:59:00 | 430 / 3.1644 | 16004 / 16004 MiB | RUNNING |

The 5-minute tail includes step 10/loss 12.8648. Latest learning rate at step 430 was 0.00019910259158012808. Last snapshot recorded utilization 63%/72%, available RAM 251261685760 bytes and available disk 2838634311680 bytes. No traceback, OOM, nonfinite loss or rank failure appeared in the captured tails. This is an early observation, not a claim that the complete log or future execution is error-free. The controller's `WAITING_OR_UNKNOWN` state accompanies a RUNNING Runner attempt and represents waiting on that active attempt.

At the last snapshot the controller's last charged amount was 1847.4 seconds: approximately 106152.6 seconds remain in the shared train/normal group and 142152.6 seconds in the total, before subsequent elapsed time and cleanup reserve. Approximate charged GPU-device time for this two-GPU attempt is 3694.8 of 280800 seconds. The ledger is periodically refreshed, so these are snapshot values, not current balances or a guarantee. S3, S4, CPU and future stages have not been run. Final Adapter and formal metrics are not yet available.

Server paths, under `/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d`: `formal-run/plan.json`, `formal-run/state.json`, `formal-run/controller.log`, `jobs/letter-a0-beauty-formal-beauty-a0-seed42-formal-train-a0/{request.json,accepted.json,status.json,work.log,supervisor.log}`. The registry snapshot comes from `/data/bwwu/lc-rec/a0-beauty-20261115/formal-experiment-registry/`. Runtime files remain on the server. Log observations are bounded, non-atomic copies; `MANIFEST.sha256` records their copied bytes.

To resume monitoring the same task, from WSL using the existing Windows SSH connection:

```bash
/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes new-server 'PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 /data/bwwu/envs/letter-20261110/bin/python /data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/releases/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty/formal_control.py query --plan /data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/formal-run/plan.json'
```

Query refreshes the existing ledger; it does not start or resume computation. Inspect the original Runner and process identities and work log as well. Never use `start` solely because a new session begins. No installed packages, toolchains or configuration changes were made. Existing untracked files were preserved.

Independent review: a separate reviewer was requested, but reported that no file-reading or command-execution tools were available to it; it could not inspect the evidence. No independent PASS is claimed. This does not replace the external prelaunch review pinned above. No new correction rounds, reruns, or changes to the frozen runtime occurred in this launch task.
