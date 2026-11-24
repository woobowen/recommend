"""Derive the stage table from captured event/CSV tails and process metadata."""
import argparse
import base64
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
LOSS = ("train/loss_step", "train/loss_epoch")
VAL = ("val/recall@5", "val/recall@10", "val/ndcg@5", "val/ndcg@10")
ERROR = re.compile(r"Traceback|out of memory|CUDA error|non.finite|(?:^|\W)(?:nan|[-+]?inf)(?:\W|$)|exception",
                   re.I | re.M)

def file_text(info):
    if not info:
        return ""
    if info.get("prefix", {}).get("complete"):
        item = info["prefix"]
        if "base64" in item:
            return base64.b64decode(item["base64"], validate=True).decode()
    tail = info.get("tail", {})
    content = tail.get("text", "")
    if tail.get("offset", 0):
        _, _, content = content.partition("\n")
    return content

def records(text):
    return [json.loads(line) for line in text.splitlines(keepends=True) if line.endswith("\n")]

def csv_rows(text):
    if not text.startswith("epoch,") and not text.startswith("step,"):
        return []
    whole_rows = [line for line in text.splitlines(keepends=True) if line.endswith("\n")]
    return list(csv.DictReader(io.StringIO("".join(whole_rows))))

def finite(mapping, names):
    return all(math.isfinite(float(mapping[name])) for name in names if mapping.get(name, "") != "")

def time_at(timestamp, tz=dt.timezone.utc):
    return dt.datetime.fromtimestamp(timestamp, tz).isoformat(timespec="milliseconds")

def analyze(s, prev, request, base_time):
    files = s["files"]
    events = records(file_text(files["output/run/events.jsonl"]))
    csv_file = next((v for k, v in files.items() if k.endswith("/metrics.csv")), None)
    rows = csv_rows(file_text(csv_file))
    loss = {key: next(({k: row[k] for k in ("step", "epoch", key)} for row in reversed(rows)
                       if row.get(key) not in (None, "")), None) for key in LOSS}
    val_events = [event for event in events if event["event"] == "validation_end"]
    validation_rows = [{k: row[k] for k in ("step", "epoch", *VAL) if row.get(k) not in ("", None)}
                       for row in rows if any(row.get(k) not in ("", None) for k in VAL)]
    last_batch = next((e for e in reversed(events) if e["event"] == "train_batch"), None)
    last = events[-1] if events else None
    phase = "initializing"
    if last:
        phase = {
            "train_start": "training",
            "train_batch": "training",
            "validation_start": "validation",
            "validation_contract": "validation",
            "validation_end": "validation completed; checkpoint/transition",
            "test_start": "Normal testing",
            "test_batch": "Normal testing",
            "test_end": "Normal testing completed",
            "error": "exception",
        }.get(last["event"], last["event"])
    if files["output/run/stage_result.json"]:
        phase = "result written"
    if s["status"]["status"] != "RUNNING":
        phase = s["status"]["status"]
    uuid = request["env"]["CUDA_VISIBLE_DEVICES"]
    gpu_line, = [line.split(", ") for line in s["gpu"]["stdout"].splitlines() if f", {uuid}, " in line]
    apps = [line.split(", ") for line in s["gpu_compute_apps"]["stdout"].splitlines() if line]
    owned = {str(p["pid"]) for p in s["status"]["workgroup_observation"]["owned_live"]}
    assert all(app[0] == uuid for app in apps if app[1] in owned), "task on an unexpected GPU"
    log_keys = ("work.log", "supervisor.log", "output/run/events.jsonl")
    outputs = {key: files[key] for key in log_keys}
    outputs["metrics.csv"] = csv_file
    sizes = {key: (item["bytes"] if item else None) for key, item in outputs.items()}
    mtimes = {key: (item["mtime_utc"] if item else None) for key, item in outputs.items()}
    delta = {key: (value - (prev["log_bytes"][key] or 0) if prev and value is not None else None)
             for key, value in sizes.items()}
    anomalies = [line for key in log_keys if key != "output/run/events.jsonl"
                 for line in file_text(files[key]).splitlines() if ERROR.search(line)]
    anomalies += [e for e in events if e["event"] == "error"]
    if not all(finite(row, (*LOSS, *VAL)) for row in rows):
        anomalies.append("non-finite persisted Lightning metric")
    if any(e["batches"] != 3097 or not math.isfinite(e["monitor"]) for e in val_events):
        anomalies.append("unexpected validation coverage or monitor")
    if min(s["free_bytes"], s["observation"]["local_staging_free_bytes"],
           s["observation"]["windows_downloads_free_bytes"]) < 2 * 1024**3:
        anomalies.append("less than 2 GiB free")
    assert s["accepted"]["request_sha256"] == s["attempt"]["request_sha256"] == s["status"]["request_sha256"]
    assert len(s["queued_formal_requests"]) == len(s["accepted_formal_requests"]) == 1
    elapsed = s["server_epoch"] - base_time
    checkpoint_lines = [line for line in file_text(files["work.log"]).splitlines()
                        if ("reached" in line and "best" in line) or "saving model" in line]
    return dict(
        point=s["observation"], phase=phase, last_event=last, last_train_batch=last_batch,
        losses=loss, loss_csv_mtime=csv_file and csv_file["mtime_utc"], validation_events=val_events,
        validation_rows=validation_rows, csv_columns=list(rows[0]) if rows else None,
        checkpoint_files=s["checkpoints"], checkpoint_log=checkpoint_lines,
        gpu=dict(index=int(gpu_line[0]), uuid=gpu_line[1], name=gpu_line[2], used_mib=int(gpu_line[3]),
                 total_mib=int(gpu_line[4]), utilization_percent=int(gpu_line[5]),
                 task_apps=[app for app in apps if app[1] in owned]),
        log_bytes=sizes, delta_bytes=delta, log_mtimes_utc=mtimes,
        source_identity={k: s["status"][k] for k in ("worker", "supervisor", "request_sha256")},
        outputs_present={name: files["output/run/" + name] is not None
                         for name in ("candidates.jsonl", "stage_result.json", "offline_evaluation.json")},
        server_free_bytes=s["free_bytes"], consumed_gpu_device_seconds=elapsed,
        remaining_gpu_device_seconds=request["timeout_seconds"] - elapsed, anomalies=anomalies,
    )

def display(number):
    return "未记录" if number is None else str(number)

def table(points):
    header = ("| 采样 | 目标 EST / UTC | 实测 EST / UTC，偏差 | 阶段 | 事件 step / microbatch，记录 EST | "
              "CSV train/loss_step（CSV step） | 最近已完成验证及 best | GPU 1 MiB / 利用率 | "
              "work / events / CSV 字节（较上点增量） | 异常 |")
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|"]
    def clocks(pair):
        return pair["new_york"][11:23] + " / " + pair["utc"][11:23]
    phases = {"initializing": "初始化", "training": "训练", "validation": "验证",
              "validation completed; checkpoint/transition": "验证已完成，保存／切换阶段"}
    for point in points:
        p = point["point"]
        event = point["last_train_batch"]
        step = f"{event['step']} / {event['microbatch']}，{time_at(event['time'], NY)[11:23]}" if event else "未产生"
        loss = point["losses"]["train/loss_step"]
        loss_note = f"{loss['train/loss_step']}（{loss['step']}）" if loss else "未记录"
        val = point["validation_events"][-1] if point["validation_events"] else None
        best = max(point["validation_events"], key=lambda e: e["monitor"]) if val else None
        vtext = (f"{val['step']} / {val['batches']} 批，{val['monitor']:.9g}，{time_at(val['time'], NY)[11:23]}"
                 f"，best {best['step']} / {best['monitor']:.9g}" if val else "未完成")
        log = point["log_bytes"]
        delta = point["delta_bytes"]
        groups = []
        for key in ("work.log", "output/run/events.jsonl", "metrics.csv"):
            groups.append(display(log[key]) + (f" ({delta[key]:+d})" if delta[key] is not None else ""))
        values = (p["label"], clocks(p["target_time"]),
                  f"{clocks(p['measured_time'])}，{p['deviation_seconds']:+.3f} s",
                  phases.get(point["phase"], point["phase"]), step, loss_note, vtext,
                  f"{point['gpu']['used_mib']} / {point['gpu']['utilization_percent']}%",
                  " / ".join(groups), "无运行异常" if not point["anomalies"] else "见原始记录")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--request", type=Path)
    ns = parser.parse_args()
    points = []
    req = json.loads((ns.request or ns.raw / "request.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False,
                                           separators=(",", ":")).encode()).hexdigest()
    first = json.loads((ns.raw / "T00.json").read_text())
    base = dt.datetime.fromisoformat(first["observation"]["target_time"]["utc"]).timestamp()
    for minute in range(0, 31, 5):
        label = f"T{minute:02d}"
        s = json.loads((ns.raw / f"{label}.json").read_text())
        assert s["observation"]["label"] == label
        target = dt.datetime.fromisoformat(s["observation"]["target_time"]["utc"]).timestamp()
        assert abs(target - base - 60 * minute) < 1e-6
        assert abs(s["server_epoch"] - target - s["observation"]["deviation_seconds"]) < 1e-6
        assert s["status"]["request_sha256"] == fingerprint
        assert s["status"]["worker"] == first["status"]["worker"] and s["status"]["supervisor"] == first["status"]["supervisor"]
        assert s["status"]["worker_alive"] and s["status"]["supervisor_alive"]
        assert s["status"]["status"] == "RUNNING"
        points.append(analyze(s, points[-1] if points else None, req, base))
    ns.output.mkdir(parents=True, exist_ok=True)
    (ns.output / "samples-derived.json").write_text(json.dumps(points, indent=2, ensure_ascii=False) + "\n")
    (ns.output / "timeline.md").write_text(table(points))

if __name__ == "__main__":
    main()
