#!/usr/bin/env python3
"""Independent small-result consistency audit. No primary/summarize imports."""
import argparse, datetime as DT, hashlib, itertools, json, math
from pathlib import Path

RUN = "f0242b9f90ae23a4979d688a287734ca9f88cea1"
STAT_SHA = "c17e47746751146eb4c50245e88616811981031dbd64501eb205e1615774ab96"
N, SID_COUNT = 24772, 9897
KEYS = ([f"hit@{k}" for k in (1,5,10,20)]
        + [f"ndcg@{k}" for k in (5,10,20)]
        + [f"mrr@{k}" for k in (5,10,20)]
        + [f"ItemHit@{k}" for k in (5,10,20)]
        + [f"ItemNDCG@{k}" for k in (5,10,20)])
JOBS = [
 ("a0",3,"letter-a0-20261112-shuffle3-instruments-seed42-20261114-r0"),
 ("a0",4,"letter-a0-20261112-shuffle4-instruments-seed42-20261114-r0"),
 ("a1",3,"letter-a1-q06-20261113-shuffle3-instruments-seed42-20261114-r0"),
 ("a1",4,"letter-a1-q06-20261113-shuffle4-instruments-seed42-20261114-r0")]
NORMAL = {
 "a0":"lc_rec/letter/evidence/final_audit_20261113/jobs/"
      "letter-a0-20261112-final-audit-20261113-offline-r1/results/primary",
 "a1":"lc_rec/a1_qwen3_06b/evidence/status_checks/20261114T034030Z/"
      "existing_results/offline_evaluation/primary"}
COUNTS = {"history_group":{"0-5":12556,"6-10":8652,"11-20":3564},
          "popularity_group":{"UNSEEN":35,"LONG_TAIL":3990,"MID":4132,"POPULAR":16615}}
FILES = ["evaluation/primary/metrics.json","evaluation/primary/groups.json",
 "evaluation/primary/bootstrap.json","evaluation/paired/paired_comparison.json",
 "evaluation/independent_inputs.json","evaluation/independent_extended.json",
 "evaluation/independent_statistics.json","evaluation/summary.json",
 "evaluation/prediction_manifest.json","test/rank0.json","test/rank1.json",
 "job/status.json","summary.json"]
READ = {}

def need(ok, message):
    if not ok: raise AssertionError(message)
def eq(a,b,message):
    need(math.isfinite(float(a)) and math.isfinite(float(b))
         and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),
         f"{message}: {a!r} != {b!r}")
def digest(path):
    p=Path(path); before=p.stat(); h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""): h.update(block)
    after=p.stat()
    need((before.st_size,before.st_mtime_ns,before.st_ino)==
         (after.st_size,after.st_mtime_ns,after.st_ino),f"changed during read: {p}")
    result={"path":str(p.resolve()),"bytes":after.st_size,"sha256":h.hexdigest()}
    READ[str(p.resolve())]=result
    return result
def load(path):
    p=Path(path); before=p.stat()
    with p.open(encoding="utf-8") as f: result=json.load(f)
    record=digest(p)
    need((before.st_size,before.st_mtime_ns,before.st_ino)==
         (p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino),
         f"JSON changed during audit: {p}")
    return result
def issha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)
def instant(value):
    result=DT.datetime.fromisoformat(value.replace("Z","+00:00"))
    need(result.tzinfo is not None,"timestamp without timezone")
    return result
def rehash_collected(folder,receipt,required):
    need(issha(receipt["sha256"]) and receipt["bytes"]>0,"invalid file receipt")
    candidates=[p for p in folder.rglob(Path(receipt["path"]).name) if p.is_file()]
    matches=[]
    for p in candidates:
        if p.stat().st_size==receipt["bytes"]:
            actual=digest(p)
            if actual["sha256"]==receipt["sha256"]: matches.append(str(p))
    if candidates or required:
        need(bool(matches),f"no collected file matches receipt: {receipt['path']}")
    return matches

def verify(folder,model,n,job,normal,repo):
    docs={name:load(folder/name) for name in FILES}
    p=docs[FILES[0]]; g=docs[FILES[1]]; b=docs[FILES[2]]
    paired=docs[FILES[3]]; inp=docs[FILES[4]]; ext=docs[FILES[5]]
    stat=docs[FILES[6]]; ev=docs[FILES[7]]; manifest=docs[FILES[8]]
    ranks=[docs[FILES[9]],docs[FILES[10]]]
    state=docs[FILES[11]]; top=docs[FILES[12]]
    nm,ng=normal
    need(p["status"]=="PASS" and p["users"]==N,"primary status/users")
    need(p["run_commit"]==p["evaluation_commit"]==RUN,"primary RUN identity")
    need(set(p["metrics"])==set(KEYS),"primary metric keys")
    need(p["gpu_used"] is False,"CPU metrics GPU flag")
    need(issha(p["user_axis_sha256"]) and
         p["user_axis_sha256"]==nm["user_axis_sha256"],"Normal/Shuffle user axis")
    need(p["protocol_version"]==nm["protocol_version"],"protocol changed")

    need(g["q1"]==ng["q1"]==5 and g["q2"]==ng["q2"]==11,"terciles")
    need(g["training_visible_interactions"]==ng["training_visible_interactions"]==156609,
         "training-visible interaction census")
    for family,names in COUNTS.items():
        for name,count in names.items():
            sg=g["groups"][family][name]; old=ng["groups"][family][name]
            need(sg["users"]==old["users"]==count,f"group count {family}/{name}")
            need(set(sg["metrics"])==set(KEYS),f"group metrics {family}/{name}")
        for key in KEYS:
            value=math.fsum(g["groups"][family][name]["metrics"][key]*count
                           for name,count in names.items())/N
            eq(value,p["metrics"][key],f"weighted group mean {family}/{key}")

    histogram=p["first_target_rank_users"]
    need(set(histogram)=={str(k) for k in range(1,21)}|{"miss"},"rank histogram keys")
    need(all(type(v) is int and v>=0 for v in histogram.values())
         and sum(histogram.values())==N,"rank histogram counts")
    for k in (1,5,10,20):
        eq(sum(histogram[str(i)] for i in range(1,k+1))/N,
           p["metrics"][f"hit@{k}"],f"first-hit HR@{k}")
    for k in (5,10,20):
        s=p["sid_statistics"][str(k)]; total=N*k
        need(s["candidates"]==total and s["valid"]+s["invalid"]==total,"SID conservation")
        need(0<=s["colliding"]<=s["valid"] and 0<=s["duplicates"]<=total,"SID count bounds")
        need(s["coverage_denominator"]==SID_COUNT
             and 0<=s["unique_valid_sids"]<=min(SID_COUNT,s["valid"]),"coverage census")
        eq(s["unique_valid_sid_coverage"],s["unique_valid_sids"]/SID_COUNT,"coverage")
        for count,rate in (("valid","valid_rate"),("invalid","invalid_rate"),
                           ("duplicates","duplicate_rate"),("colliding","collision_candidate_rate")):
            eq(s[rate],s[count]/total,f"SID {k} {rate}")
        for count,users,rate in (("invalid","users_invalid","users_invalid_rate"),
              ("duplicates","users_duplicate","users_duplicate_rate"),
              ("colliding","users_colliding","users_colliding_rate")):
            need(0<=s[users]<=min(N,s[count]) and s[count]<=s[users]*k,"SID affected users")
            eq(s[rate],s[users]/N,f"SID {k} {rate}")
    cat=p["collision_catalog"]
    for key,value in {"items":9922,"unique_sids":9897,"groups":23,
                      "affected_items":48,"maximum_group_size":3,
                      "target_collision_users":122}.items():
        need(cat[key]==value,f"collision census {key}")
    need(cat["group_size_distribution"]=={"2":21,"3":2},"collision sizes")

    need(paired["status"]=="PASS" and paired["users"]==N,"paired status/users")
    need(len(paired["identity_fields_exact"])==12,"paired identity field census")
    need(set(paired["metrics"])==set(KEYS),"paired metric keys")
    for key in KEYS:
        x=paired["metrics"][key]
        eq(x["Normal"],nm["metrics"][key],f"Normal point {key}")
        eq(x["Shuffle"],p["metrics"][key],f"Shuffle point {key}")
        delta=x["Shuffle"]-x["Normal"]; eq(x["absolute_difference"],delta,f"delta {key}")
        if x["Normal"]==0: need(x["relative_difference"] is None,"zero Normal relative difference")
        else: eq(x["relative_difference"],delta/x["Normal"],f"relative delta {key}")
        need(math.isfinite(x["difference_CI95_low"]) and math.isfinite(x["difference_CI95_high"])
             and x["difference_CI95_low"]<=x["difference_CI95_high"],f"paired CI {key}")
    pb=paired["bootstrap"]
    need(pb["repetitions"]==5000 and pb["seed"]==20261113 and pb["rng"]=="PCG64"
         and pb["replacement"] is True and pb["percentile"]=="95% linear","paired bootstrap settings")
    iv=paired["independent_verification"]
    need(iv["source_sha256"]==STAT_SHA and 0<iv["tolerance"]<=1e-13,"paired verifier identity")
    need(0<=iv["maximum_draw_difference"]<=iv["tolerance"] and
         0<=iv["maximum_interval_difference"]<=iv["tolerance"],"paired independent numeric verification")
    changes=paired["changes"]; need(changes["users"]==N,"change user census")
    for name in ("top1_changed","top20_ordered_changed","top20_set_changed"):
        need(type(changes[name]) is int and 0<=changes[name]<=N,f"change count {name}")
        eq(changes[name+"_fraction"],changes[name]/N,f"change fraction {name}")
    need(changes["top1_changed"]<=changes["top20_ordered_changed"] and
         changes["top20_set_changed"]<=changes["top20_ordered_changed"],"recommendation change containment")
    for k in (1,5,10,20):
        key=f"hit@{k}"; t=changes["hit_transitions"][key]
        need(set(t)=={"0->0","0->1","1->0","1->1"} and
             all(type(v) is int and v>=0 for v in t.values()) and sum(t.values())==N,"transition census")
        eq((t["1->0"]+t["1->1"])/N,nm["metrics"][key],f"transition Normal {key}")
        eq((t["0->1"]+t["1->1"])/N,p["metrics"][key],f"transition Shuffle {key}")
        eq((t["0->1"]-t["1->0"])/N,paired["metrics"][key]["absolute_difference"],f"transition delta {key}")

    need(b["unit"]=="user" and b["replacement"] is True and b["repetitions"]==5000
         and b["seed"]==20261113 and b["rng"]=="numpy Generator PCG64"
         and b["percentiles"]==[2.5,97.5] and b["percentile_method"]=="linear"
         and b["users"]==N,"primary bootstrap settings")
    fs=stat["full_statistics"]
    need(stat["status"]==fs["status"]=="PASS" and stat["gpu_used"] is False,"statistics receipt")
    need(fs["users_compared"]==N and fs["replicate_means_compared"]==80000
         and 0<fs["absolute_tolerance"]<=1e-13,"statistics extent")
    for key in ("bootstrap_max_draw_absolute_difference","bootstrap_max_interval_absolute_difference",
                "bootstrap_max_point_absolute_difference","subgroup_max_absolute_difference"):
        need(0<=fs[key]<=fs["absolute_tolerance"],f"statistics delta {key}")
    need(fs["source_identity"]["sha256"]==STAT_SHA,"statistics source identity")
    for key in KEYS:
        x=b["intervals"][key]; eq(x["estimate"],p["metrics"][key],f"bootstrap point {key}")
        need(math.isfinite(x["low"]) and math.isfinite(x["high"]) and x["low"]<=x["high"],"bootstrap CI order")
        for field in ("estimate","low","high"):
            eq(fs["intervals"][key][field],x[field],f"independent bootstrap {key}/{field}")
    local_rehash={}
    for key in ("primary_bootstrap","primary_draws","primary_groups","primary_metric_array"):
        local_rehash[key]=rehash_collected(folder,fs["input_identity"][key],True)
    need(b["metric_array_sha256"]==fs["input_identity"]["primary_metric_array"]["sha256"],"metric-array identity")

    need(ext["status"]=="PASS" and ext["run_commit"]==RUN and ext["actual_test_users"]==N
         and ext["exit_code"]==0 and issha(ext["script_sha256"]),"raw independent receipt")
    pc=ext["per_user_comparison"]
    need(pc["checked_rows"]==N and pc["checked_metric_values"]==N*16
         and 0<pc["absolute_tolerance"]<=1e-14,"raw independent comparison extent")
    ds=pc["maximum_absolute_delta_by_metric"]
    need(len(ds)==16 and all(0<=v<=pc["absolute_tolerance"] for v in ds.values()),"raw independent deltas")

    need(inp["status"]=="PASS_INDEPENDENT_INPUT_REPLAY" and inp["users"]==N
         and inp["decoded_candidates_checked"]==N*20 and inp["seed"]==42
         and inp["shuffle_n"]==n and inp["protocol"]=="GRID-LETTER-SHUFFLE-V1"
         and inp["input_changed_fraction"]==inp["nonidentity_fraction"]==1,"input replay receipt")
    need(inp["history_groups"]==172929,"history group census")
    frequencies=inp["permutation_frequencies"]
    observed=set()
    for encoded,count in frequencies.items():
        perm=json.loads(encoded)
        need(isinstance(perm,list) and len(perm)==n
             and all(type(i) is int for i in perm)
             and sorted(perm)==list(range(n)) and perm!=list(range(n))
             and type(count) is int and count>=0,"illegal/identity permutation")
        observed.add(tuple(perm))
    catalog=set(itertools.permutations(range(n)))-{tuple(range(n))}
    need(len(catalog)==(5 if n==3 else 23)
         and observed==catalog and len(frequencies)==len(catalog),
         "incomplete/duplicate nonidentity permutation frequency catalog")
    need(sum(frequencies.values())==172929,"permutation frequency census")
    need(manifest["status"]=="FROZEN_INPUTS" and manifest["model_key"]==model
         and manifest["task_schema"]=="letter_shuffle_frozen_prediction_v1"
         and manifest["gpu_run_commit"]==RUN,"prediction identity receipt")
    preds=manifest["predictions"]
    need(len(preds)==2 and {x["rank"] for x in preds}=={0,1},"prediction ranks")
    input_shas={x["sha256"] for x in inp["files"]}
    audit_path=repo/"lc_rec/shuffle/evidence/cpu-bound-r0"/model/"audit.json"
    cpu_audit=load(audit_path)
    cpu_audit_identity=digest(audit_path)
    audit_matches=[
        x for x in inp["files"]
        if x["sha256"]==cpu_audit_identity["sha256"]
        and x["bytes"]==cpu_audit_identity["bytes"]
    ]
    need(len(audit_matches)==1,"CPU audit bytes/SHA not uniquely bound to input receipt")
    permutation_manifest_sha=cpu_audit["manifest"]["sha256"]
    need(issha(permutation_manifest_sha),"invalid CPU permutation manifest SHA")
    for x in preds:
        need(issha(x["sha256"]) and x["bytes"]>0 and x["sha256"] in input_shas,"prediction/input receipt binding")

    need(ev["status"]=="PASS" and ev["INPUT_PREDICTION_RUN_COMMIT"]==
         ev["EVALUATION_RUN_COMMIT"]==RUN and ev["model_key"]==model
         and ev["shuffle_n"]==n and ev["GPU_used"] is False
         and ev["normal_prediction_CPU_regression_fixture"] is False,"formal evaluation identity")
    need([x["stage"] for x in ev["stages"]]==["independent_inputs","independent_original",
         "primary","independent_extended","independent_statistics","paired"],"CPU stage sequence")
    need(all(x["exit_code"]==0 and x["wall_seconds"]>=0 for x in ev["stages"]),"CPU stage exits")
    need(top["status"]=="PASS" and top["job_id"]==job and top["model_key"]==model
         and top["shuffle_n"]==n and top["RUN_COMMIT"]==RUN
         and top["generation_exit_code"]==top["evaluation_exit_code"]==0,"top-level result")
    for rank,r in enumerate(ranks):
        need(r["status"]=="PASS" and r["rank"]==r["local_rank"]==rank
             and r["world_size"]==2 and r["completed_users"]==12386
             and r["RUN_COMMIT"]==RUN and r["preflight_only"] is False,"rank completion")
        need(issha(r["entry_sha256"]) and issha(r["allowed_callback_sha256"]),"rank source identities")
        s=r["shuffle"]; need(s["n"]==n and s["seed"]==42 and s["protocol"]=="GRID-LETTER-SHUFFLE-V1"
             and s["no_explicit_old_positions_or_prefilled_cache"] is True
             and s["cpu_manifest_sha256"]==permutation_manifest_sha,"rank shuffle/source binding")
        generation=r["generation"]
        for key,value in {"prompt_id":0,"num_beams":20,"num_return_sequences":20,
                          "max_new_tokens":10,"per_rank_batch_size":1}.items():
            need(generation[key]==value,f"generation condition {key}")
        need(generation["early_stopping"] is True and generation["filter_items"] is False
             and generation["oom_beam_reduction"] is False,"generation switches")
        need(0<=r["gpu_peak_allocated_bytes"]<=r["gpu_peak_reserved_bytes"]<=24*1024**3,"VRAM bounds")
    need(ranks[0]["entry_sha256"]==ranks[1]["entry_sha256"] and
         ranks[0]["allowed_callback_sha256"]==ranks[1]["allowed_callback_sha256"] and
         ranks[0]["shuffle"]["cpu_manifest_sha256"]==ranks[1]["shuffle"]["cpu_manifest_sha256"],
         "rank source/manifest agreement")
    need(state["status"]=="SUCCEEDED" and state["exit_code"]==0
         and state["job_id"]==job and state["worker_may_be_live"] is False
         and state["cleanup"]["verified"] is True,"supervisor/cleanup")
    observation=state["cleanup"]["last_observation"]
    need(observation["verified_empty"] is True and all(not observation[k] for k in
         ("owned_live","unsupported_live","unconfirmed_live","scan_errors")),"cleanup observation")
    start,end=instant(state["started_at"]),instant(state["ended_at"])
    need(start<=instant(top["started_utc"])<=instant(top["ended_utc"])<=end,"job timestamp order")
    gpu=top["gpu_stage_seconds"]; cpu=top["cpu_evaluation_seconds"]
    controlled=max(gpu,(end-start).total_seconds()-cpu)
    limit=14400 if model=="a0" else 10800
    need(0<gpu<=limit and 0<controlled<=limit and cpu>=0,"per-condition actual GPU budget")
    eq(top["gpu_device_seconds"],2*gpu,"GPU-device accounting")
    return {"status":"PASS","job":job,"model":model,"shuffle_n":n,
        "started":start.isoformat(),"ended":end.isoformat(),"controlled_wall_seconds":controlled,
        "gpu_stage_seconds":gpu,"gpu_device_seconds":top["gpu_device_seconds"],
        "user_axis_sha256":p["user_axis_sha256"],"history_groups":inp["history_groups"],
        "cpu_audit_identity":cpu_audit_identity,
        "permutation_manifest_sha256":permutation_manifest_sha,
        "permutation_manifest_locally_rehashed":False,
        "history_lengths":inp["history_lengths"],"permutation_frequencies":frequencies,
        "local_statistics_rehashed":local_rehash,"predictions":preds}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--repo")
    parser.add_argument("--out")
    args=parser.parse_args()
    root=Path(args.root).resolve(); repo=Path(args.repo).resolve() if args.repo else root.parents[2]
    results=[]; pending=[]; errors=[]; normals={}
    for model,n,job in JOBS:
        folder=root/job
        missing=[name for name in FILES if not (folder/name).is_file()]
        if missing:
            pending.append({"job":job,"missing":missing}); continue
        try:
            if model not in normals:
                base=repo/NORMAL[model]
                normals[model]=(load(base/"metrics.json"),load(base/"groups.json"))
            results.append(verify(folder,model,n,job,normals[model],repo))
        except Exception as exc:
            errors.append({"job":job,"error":f"{type(exc).__name__}: {exc}"})
    if len(results)==4 and not errors:
        try:
            need(len({r["user_axis_sha256"] for r in results})==1,"four-condition user axis")
            need(len({r["history_groups"] for r in results})==1,"four-condition history groups")
            need(all(r["history_lengths"]==results[0]["history_lengths"] for r in results),"history lengths")
            for n in (3,4):
                pair=[r for r in results if r["shuffle_n"]==n]
                need(pair[0]["permutation_frequencies"]==pair[1]["permutation_frequencies"],
                     f"cross-model fixed-permutation frequency identity S{n}")
            for before,after in zip(results,results[1:]):
                need(instant(before["ended"])<=instant(after["started"]),"formal serial/cleanup ordering")
            need(sum(r["controlled_wall_seconds"] for r in results)<=18*3600,
                 "18-hour cumulative formal GPU/startup/cleanup budget")
        except Exception as exc: errors.append({"cross_condition":str(exc)})
    report={"status":"FAIL" if errors else "PENDING" if pending else "PASS",
        "scope":"independent small-result algebra, identity, receipt and actual-budget consistency",
        "not_a_raw_prediction_rescan":True,"not_final_research_audit":True,
        "RUN_COMMIT":RUN,"source":digest(__file__),"conditions":results,
        "pending":pending,"errors":errors,"local_files_rehashed":list(READ.values()),
        "formal_controlled_wall_seconds":sum(r["controlled_wall_seconds"] for r in results),
        "gpu_device_seconds":sum(r["gpu_device_seconds"] for r in results),
        "limitations":["Raw JSONL/CSV stay on server; their complete scans use recorded independent runs.",
          "NPZ/NPY contents are not re-evaluated here; collected bytes are rehashed against receipts.",
          "Per-user permutation replay/conservation uses independent input-checker execution receipts.",
          "Preflight GPU budget and full asset hash receipts remain separate audit evidence."]}
    text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    if args.out:
        with Path(args.out).open("x",encoding="utf-8") as f:f.write(text+"\n")
    print(text)
    return 1 if errors else 2 if pending else 0

if __name__=="__main__":
    raise SystemExit(main())
