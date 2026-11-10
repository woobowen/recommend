"""Independent local verification of existing CPU artifacts; no job is rerun."""
import collections,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
JOB=ROOT/"evidence/jobs/letter-20261110-data-r1"
OUT=JOB/"output/audit"
review=ROOT/"review/identity"
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
audit=read(OUT/"audit.json");manifest=read(JOB/"export-manifest.json")
failures=[]
for member in manifest["files"]:
    p=JOB/member["path"]
    if p.stat().st_size!=member["bytes"] or sha(p)!=member["sha256"]: failures.append(member["path"])
assert not failures,failures
for relative,entry in audit["output_files"].items():
    p=OUT/relative
    assert p.stat().st_size==entry["bytes"] and sha(p)==entry["sha256"],relative
status=read(JOB/"status.json");query=read(JOB/"last-query.json");request=read(JOB/"request.json")
assert status["status"]==query["status"]=="WORKERS_TERMINATED"
assert status["exit_code"]==query["exit_code"]==0
assert query["cleanup_verified"] and query["workgroup_observation"]["verified_empty"]
assert not query["worker_alive"] and not query["supervisor_alive"] and not query["worker_may_be_live"]
assert canon(request)==query["request_sha256"]
assert audit["script_sha256"]==sha(ROOT/"scripts/audit_data.py")
for name,record in audit["original_source"].items():
    assert record["sha256"]==sha(ROOT/"vendor/LETTER/LETTER-LC-Rec"/name)
for record in request["inputs"]:
    rel=record["path"].split("/lc_rec/letter/",1)[1]
    assert sha(ROOT/rel)==record["sha256"]
a_idx=read(ROOT/"vendor/LETTER/data/Instruments/Instruments.index.json")
inters=read(ROOT/"vendor/LETTER/data/Instruments/Instruments.inter.json")
records={"train":[],"valid":[],"test":[]}
for ordinal,(uid,seq) in enumerate(inters.items()):
    for mode in records:
        positions=range(1,len(seq)-2) if mode=="train" else [len(seq)-(2 if mode=="valid" else 1)]
        for position in positions:
            records[mode].append({"user_ordinal":ordinal,"user_id":uid,"target_position":position,
                "history_item_ids":list(map(str,seq[max(0,position-20):position])),
                "target_item_id":str(seq[position])})
arrays=np.load(OUT/"length_rows.npz",allow_pickle=False)
matrices={}
for mode,rows in records.items():
    hist=np.array([len(r["history_item_ids"]) for r in rows])
    assert np.array_equal(hist,arrays[mode+"_history_lengths"])
    matrices[mode]={kind:arrays[mode+"_"+kind+"_lengths"] for kind in ("input","full")}
    assert matrices[mode]["input"].shape==(len(rows),17)
    assert np.all(matrices[mode]["full"]-matrices[mode]["input"]==5)
    labels=[[r["user_id"],r["target_position"],r["target_item_id"]] for r in rows]
    assert canon(labels)==audit["comparison"]["split_labels"][mode]["a_labels_sha256"]
    for kind,tablekey,distkey in [("input","per_class_input_lengths","input_per_prompt"),
                                  ("full","per_class_full_lengths_including_eos","full_with_eos_per_prompt")]:
        table=audit["length_measurement"][tablekey]
        expected=np.array([table[str(h)] for h in hist])
        assert np.array_equal(matrices[mode][kind],expected)
        for prompt in range(17):
            values=matrices[mode][kind][:,prompt];vals,counts=np.unique(values,return_counts=True)
            dist=audit["length_distributions"][mode][distkey][prompt]
            assert {str(int(x)):int(n) for x,n in zip(vals,counts)}==dist["histogram"]
            assert abs(float(values.mean())-dist["mean"])<1e-12
rng=np.random.RandomState(42)
valid_ids=np.stack([rng.choice(17,2,replace=False) for _ in records["valid"]])
assert np.array_equal(valid_ids,arrays["valid_prompt_ids_seed42"])
probe=read(OUT/"probe_samples.json")
for label,field,mode in [("train","train_representatives","train"),("test0","test_prompt0_representatives","test"),("testall","test_all_prompt_representatives","test")]:
    for name,sample in probe[field].items():
        row=records[mode][sample["row_idx"]]
        assert all(sample[k]==row[k] for k in row)
        p=sample["prompt_id"]
        assert sample["input_length"]==int(matrices[mode]["input"][sample["row_idx"],p])
        assert sample["full_length_including_eos"]==int(matrices[mode]["full"][sample["row_idx"],p])
assert len(probe["train_mixed"])==len({x["row_idx"] for x in probe["train_mixed"]})==128
tok_a=read(OUT/"tokenizer_a_ids.json");tok_b=read(OUT/"tokenizer_b_ids.json")
tokens=sorted({t for sid in a_idx.values() for t in sid})
assert tok_a["token_to_id"]=={t:32000+i for i,t in enumerate(tokens)}
assert read(OUT/"tokenizer_a/added_tokens.json")==tok_a["token_to_id"]
for item in audit["mask_audit"]["checks"]:
    row=records["train"][item["row_idx"]]
    expected=[tok_a["token_to_id"][t] for t in a_idx[row["target_item_id"]]]+[2]
    assert item["supervised_ids"]==expected and item["supervised_count"]==5 and item["mask_exact_match"]
remote=read(review/"remote_cpu_identity.json")
for kind,value in remote["assets"].items():
    assert value==audit["assets"]["b"]["identity"][kind]
assert remote["remote_request"]["sha256"]==sha(JOB/"request.json")
assert remote["remote_request"]["canonical_sha256"]==query["request_sha256"]
for mode in records:
    assert remote["split_labels"][mode]["labels_sha256"]==audit["comparison"]["split_labels"][mode]["b_labels_sha256"]
    assert remote["split_labels"][mode]["colliding_target_rows"]==audit["split_sid_identity"]["b"][mode]["target_rows_with_nonunique_sid"]
result={"status":"PASS","kind":"Independent consistency verification of existing data-r1 artifacts; no tokenizer job or model rerun",
        "manifest_files_verified":len(manifest["files"]),"audit_output_files_verified":len(audit["output_files"]),
        "request_bytes_match_server_and_export_manifest":True,"original_source_hashes_verified":len(audit["original_source"]),
        "local_script_hash_matches_executed":True,"request_input_hashes_verified":len(request["inputs"]),
        "npz_rows_and_17_prompt_class_tables_verified":{k:len(v) for k,v in records.items()},
        "all_per_prompt_histograms_and_means_verified":True,"exact_validation_randomstate42_verified":True,
        "real_probe_rows_verified":True,"mask_sid_plus_eos_verified":True,"a_token_ids_lexical32000_verified":True,
        "server_b_identity_fresh_matches_cpu_audit":True,"runner_state":"WORKERS_TERMINATED","leader_exit_code":0,
        "cleanup_verified":True,"remaining_task_members":0,"gpu_evidence_reviewed":False,
        "source_script_sha256":audit["script_sha256"],"audit_sha256":sha(OUT/"audit.json")}
(review/"cpu_artifact_verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

