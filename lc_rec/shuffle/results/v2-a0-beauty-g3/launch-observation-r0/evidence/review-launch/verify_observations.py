"""Verify recorded G3 formal launch observation sequence without remote or model work."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

PARENT = "g3-beauty-v2-group-formal-r0"
CHILD = "g3-beauty-v2-gpu-formal-shuffle2-r0"
GPU = "86660be5a62c8325caf74cf2810581850ca12803"
TRAINING = "21d495b0be5b606853ec7cec68a3eaa1b58acbd7"
MODEL = "3315622866cd97833741608ca21d82b72a10184ab8bff4dc688892146d0bf915"
FULL_ASSIGNMENT = "22ee58fe9f948f50e5cbc92160b6dc7df7e2f633cb9e8bf3121da5fb35bcb088"
EASTERN = ZoneInfo("America/New_York")
LIMIT = {"formal_gpu": 129600, "preflight_gpu": 10800, "cpu": 28800, "parent_wall": 126000}
PATTERN = {"formal_gpu": "g3-beauty-v2-gpu-formal-", "preflight_gpu": "g3-beauty-v2-gpu-preflight-",
           "cpu": "g3-beauty-v2-cpu-", "parent_wall": "g3-beauty-v2-group-formal-"}


def same_identity(observed, original):
    return all(observed[key] == original[key] for key in ("pid", "start_ticks", "boot_id", "uid"))


def verify(directory, startup, gpu_uuid):
    previous = None
    identities = None
    result = []
    fixed_stats = {}
    for minute in range(0, 31, 5):
        label = f"T{minute:02}"
        path = directory / f"{label}.json"
        raw = path.read_bytes()
        point = json.loads(raw)
        observed = datetime.fromisoformat(point["utc"])
        eastern = datetime.fromisoformat(point["america_new_york"])
        assert point["label"] == label and point["host"] == "zwlab-01"
        assert eastern == observed.astimezone(EASTERN) and eastern.utcoffset() == observed.astimezone(EASTERN).utcoffset()
        assert abs(observed.timestamp() - point["unix_ns"] / 1e9) <= 1e-6
        if minute == 0:
            t0 = observed
            t0_ns = point["unix_ns"]
            assert datetime.fromisoformat(point["child"]["started_at"]) <= observed
        span = (point["unix_ns"] - t0_ns) / 1e9
        assert math.isclose(span, point["elapsed_since_t0_seconds"], abs_tol=1e-6)
        interval = (point["unix_ns"] - previous["unix_ns"]) / 1e9 if previous else None
        assert interval is None or interval > 0
        rank, = point["gpu_rank_processes"]
        snapshot_identities = {f"{job}_{role}": point[job][role] for job in ("parent", "child")
                               for role in ("worker", "supervisor")}
        snapshot_identities["gpu_rank0"] = rank
        if identities is None:
            identities = snapshot_identities
        assert all(same_identity(record, identities[key]) for key, record in snapshot_identities.items())
        for group, expected in (("parent", PARENT), ("child", CHILD)):
            state = point[group]
            assert state["job_id"] == expected and state["status"] == "RUNNING"
            assert state["worker_alive"] and state["supervisor_alive"] and state["workgroup_live"]
            assert state.get("ended_at") is None and state.get("exit_code") is None
            chk = startup["authorization_and_request_materialization"][f"{group}_request_canonical_sha256"]
            assert state["request_sha256"] == chk
        assert rank["proc_parent_pid"] == point["child"]["worker"]["pid"]
        env = rank["rank_env"]
        assert {k: env[k] for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GRID_JOB_ID", "CUDA_VISIBLE_DEVICES")} == {
            "WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0", "GRID_JOB_ID": CHILD, "CUDA_VISIBLE_DEVICES": gpu_uuid}
        assert "--scope" in rank["argv"] and rank["argv"][rank["argv"].index("--scope")+1] == "FORMAL"
        assert not any("--preflight" in token for token in rank["argv"])
        assert point["controller"]["status"] == "RUNNING"
        assert [stage["job_id"] for stage in point["controller"]["stages"]] == [CHILD]
        gpu = point["gpu"]
        assert gpu["uuid"] == gpu_uuid and gpu["physical_index"] == "1"
        owned = [app for app in point["compute_apps"] if int(app["pid"]) == rank["pid"] or app["uuid"] == gpu_uuid]
        assert len(owned) <= 1
        assert all(app["uuid"] == gpu_uuid and int(app["pid"]) == rank["pid"] for app in owned)
        assert 0 <= int(gpu["utilization_pct"]) <= 100
        assert 0 <= int(gpu["memory_used_mib"]) <= int(gpu["memory_total_mib"])
        assert point["output_free_bytes"] >= 20 * 1024**3
        assert not point["error_matches"]

        prediction = point["prediction"]
        count = prediction["complete_users"]
        old_count = previous["prediction"]["complete_users"] if previous else 0
        old_bytes = previous["prediction"]["bytes"] if previous else 0
        old_complete_bytes = previous["prediction"]["complete_bytes"] if previous else 0
        if previous:
            assert count > old_count
        assert count < 22363
        assert prediction["delta_users"] == count - old_count
        assert prediction["delta_bytes"] == prediction["bytes"] - old_bytes
        assert prediction["bytes"] == prediction["complete_bytes"] + prediction["buffered_suffix_bytes"]
        if count:
            assert len(owned) == 1
            report = point["rank"]
            assert (report["scope"], report["rank"], report["local_rank"], report["world_size"]) == ("FORMAL", 0, 0, 1)
            assert report["assigned_users"] == 22363 and report["conditions"] == ["shuffle2"]
            assert report["RUN_COMMIT"] == GPU and report["MODEL_TRAINING_COMMIT"] == TRAINING
            assert report["model_identity_sha256"] == MODEL and report["assignment_sha256"] == FULL_ASSIGNMENT
            assert 0 <= count - report["completed_users"] <= 25 and report["error"] is None
            assert prediction["delta_offset_start"] == old_complete_bytes
            assert prediction["delta_complete_bytes"] == prediction["complete_bytes"] - old_complete_bytes
            if previous and previous["prediction"]["complete_users"]:
                assert prediction["inode"] == previous["prediction"]["inode"]

        last_progress = None
        for name, log in point["logs"].items():
            if log is None:
                continue
            older = (previous["logs"].get(name) or {}) if previous else {}
            assert log["delta_bytes"] == log["bytes"] - older.get("bytes", 0)
            assert log["offset_end"] == log["bytes"] and older.get("bytes", 0) <= log["offset_start"] <= log["offset_end"]
            if older:
                assert (log["device"], log["inode"]) == (older["device"], older["inode"])
                assert log["mtime_ns"] >= older["mtime_ns"]
            raw_excerpt = log["excerpt"].encode()
            if "\ufffd" not in log["excerpt"]:
                assert len(raw_excerpt) == log["offset_end"] - log["offset_start"]
                assert hashlib.sha256(raw_excerpt).hexdigest() == log["excerpt_sha256"]
            if name == "gpu_worker":
                for line in log["excerpt"].splitlines():
                    if line.startswith("{"):
                        event = json.loads(line)
                        if event.get("stage") == "generation":
                            assert event["scope"] == "FORMAL" and event["conditions"] == ["shuffle2"]
                            last_progress = event["completed_users"]
        if count:
            # Races are bounded by the 25-row report interval. The live log may
            # contain a generation count newer than the prior file fstat.
            assert last_progress is not None and abs(last_progress-count) <= 25

        for k, limit in LIMIT.items():
            ledger = point["budget"][k]
            assert ledger["limit_seconds"] == limit
            values = []
            for name, record in ledger["jobs"].items():
                assert name.startswith(PATTERN[k])
                start = datetime.fromisoformat(record["started_at"]) if record["started_at"] else observed
                end = datetime.fromisoformat(record["ended_at"]) if record["ended_at"] else observed
                used = max(0, (end-start).total_seconds())
                assert math.isclose(record["worker_seconds"], used, abs_tol=1e-6)
                values.append(used)
            used = sum(values)
            assert math.isclose(ledger["used_seconds"], used, abs_tol=1e-6)
            assert math.isclose(ledger["remaining_seconds"], limit-used, abs_tol=1e-6)
            assert 0 <= used < limit
            if k in ("preflight_gpu", "cpu"):
                if minute == 0:
                    fixed_stats[k] = used
                else:
                    assert math.isclose(used, fixed_stats[k], abs_tol=1e-6)
            elif k in ("formal_gpu", "parent_wall"):
                assert list(ledger["jobs"]) == [CHILD if k == "formal_gpu" else PARENT]
        result.append({"label": label, "utc": point["utc"], "america_new_york": point["america_new_york"],
                       "snapshot_sha256": hashlib.sha256(raw).hexdigest(), "complete_users": count,
                       "new_users": prediction["delta_users"], "latest_generation_log_users": last_progress,
                       "elapsed_seconds": span, "deviation_from_nominal_seconds": span - minute * 60,
                       "interval_since_previous_seconds": interval,
                       "formal_gpu_worker_seconds": point["budget"]["formal_gpu"]["used_seconds"],
                       "parent_wall_seconds": point["budget"]["parent_wall"]["used_seconds"]})
        previous = point
    assert span >= 1800 and len(result) == 7
    return {"status": "PASS", "scope": "same formal S2 and parent across seven observed points",
            "span_seconds": span, "t0": t0.isoformat(), "t30": previous["utc"],
            "rank_pid": identities["gpu_rank0"]["pid"], "gpu_uuid": gpu_uuid,
            "preflight_gpu_worker_seconds": fixed_stats["preflight_gpu"], "cpu_runner_worker_seconds": fixed_stats["cpu"],
            "points": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("startup_review", type=Path)
    args = parser.parse_args()
    startup = json.loads(args.startup_review.read_text())
    card = startup["authorization_and_request_materialization"]["gpu_uuid"]
    print(json.dumps(verify(args.directory, startup, card), indent=2))


if __name__ == "__main__":
    main()
