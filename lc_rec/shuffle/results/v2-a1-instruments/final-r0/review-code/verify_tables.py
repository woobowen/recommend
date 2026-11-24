#!/usr/bin/env python3
"""Cross-check publication CSV/JSON against audited conditions and server paired output."""
import argparse
from collections import Counter
import csv
from hashlib import file_digest
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def check(ok, label):
    if not ok:
        raise ValueError(label)


def verify(raw, tables, source):
    root = tables / "results.json"
    published = read(root)
    production = read(raw / "evidence/formal-paired-r0/summary.json")
    audit = read(source)
    check(audit["status"] == "PASS_NON_AUTHOR_B_FORMAL_ALL_METRICS_RECOMPUTED", "Raw candidate audit prerequisite")
    check(published["status"] == "PASS_FORMAL_PUBLICATION_TABLES" and published["users"] == production["users"] == audit["users_per_condition"],
          "Publication users and status")
    p = audit["per_condition"]
    check(published["metrics"] == production["metrics"] == {mode:r["metrics"] for mode,r in p.items()},
          "Published means differ from audited raw predictions")
    check(published["candidate_anomalies"] == {mode:r["candidate_anomalies"] for mode,r in p.items()}, "Published diagnostics")
    check(all(published[k] == production[k] for k in ("bootstrap","arrays_file_sha256","comparisons")), "Published paired summary identity")
    ordered = list(published["metrics"])
    keys = published["metric_keys"]
    check(len(keys) == len(set(keys)) == 16 and set(keys) == set(p["normal"]["metrics"]), "Publication metric order")
    metrics_csv = csv_rows(tables / "metrics.csv")
    check([row["condition"] for row in metrics_csv] == ordered, "Metric CSV condition order")
    for row in metrics_csv:
        mode = row.pop("condition")
        check(set(row) == set(keys) and {metric:float(value) for metric,value in row.items()} == p[mode]["metrics"],
              f"{mode} published metrics CSV")
    records = []
    max_percentage_arithmetic_order_difference = 0.0
    for mode in ordered[1:]:
        for metric in keys:
            a, b = p["normal"]["metrics"][metric], p[mode]["metrics"][metric]
            diff = b-a
            percent = (100*diff)/a if a else None
            if percent is not None:
                max_percentage_arithmetic_order_difference = max(
                    max_percentage_arithmetic_order_difference, abs(percent - 100*(diff/a)))
            records.append({"condition":mode,"metric":metric,"normal":a,"value":b,"proportion_difference":diff,
                            "percentage_point_difference":100*diff,"relative_change_percent":percent})
    check(published["changes_from_normal"] == records, "Published 64 delta records")
    deltas_csv = csv_rows(tables / "changes_from_normal.csv")
    check(len(deltas_csv) == len(records), "Delta CSV row count")
    for pubrow, ref in zip(deltas_csv, records, strict=True):
        row = {k:(v if k in ("condition","metric") else float(v) if v != "" else None) for k,v in pubrow.items()}
        check(row == ref, f"{ref['condition']}/{ref['metric']} published delta CSV")
    signed = {mode:dict(Counter("up" if x["proportion_difference"]>0 else "down" if x["proportion_difference"]<0 else "equal"
                               for x in records if x["condition"] == mode)) for mode in ordered[1:]}
    rankings = {metric:sorted(ordered, key=lambda v:p[v]["metrics"][metric], reverse=True) for metric in keys}
    check(published["signed_change_counts"] == signed and published["rankings_descending"] == rankings, "Published effect rankings")
    intervals = {}
    pairs = {}
    for name, rows in production["comparisons"].items():
        group = {"wholly_below_zero":[],"includes_zero":[],"wholly_above_zero":[]}
        for row in rows:
            low,high = (row[f"difference_CI95_{k}"] for k in ("low","high"))
            group["wholly_below_zero" if high < 0 else "wholly_above_zero" if low > 0 else "includes_zero"].append(row["metric"])
            pairs[name,row["metric"]] = row
        intervals[name] = group
    check(published["interval_summary"] == intervals, "Published interval direction counts")
    output_pairs = csv_rows(tables / "paired.csv")
    check(len(output_pairs) == len(pairs), "Publication paired CSV row count")
    for row in output_pairs:
        match = (row.pop("comparison"),row.pop("metric"))
        check(match in pairs, "Unique publication pair/metric")
        reference = pairs.pop(match)
        check(set(row) == (set(reference)-{"metric"}) and
              {key:float(value) if value != "" else None for key,value in row.items()} ==
              {key:val for key,val in reference.items() if key != "metric"}, f"Paired CSV row {match}")
    return {"status":"PASS_NON_AUTHOR_B_FORMAL_PUBLICATION_TABLES", "audited_condition_means":80,
            "audited_normal_change_rows":len(records),"audited_paired_rows":len(output_pairs),
            "signed_change_counts":signed,"interval_summary":intervals,
            "max_relative_percent_difference_if_dividing_before_multiplying":max_percentage_arithmetic_order_difference,
            "table_sha256":{name:sha(tables/name) for name in ("metrics.csv","changes_from_normal.csv","paired.csv","results.json")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("raw", "tables", "metrics-audit", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    a=parser.parse_args()
    result=verify(a.raw,a.tables,a.metrics_audit)
    with a.output.open("x", encoding="utf-8") as out:
        json.dump(result,out,indent=2,ensure_ascii=False,allow_nan=False)
        out.write("\n")
    print(result["status"])
