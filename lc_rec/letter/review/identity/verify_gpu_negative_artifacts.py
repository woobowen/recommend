"""Verify preserved negative GPU evidence only; never loads models or invokes CUDA."""
import collections,hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
REVIEW=ROOT/"review/identity"
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
cpu=ROOT/"evidence/jobs/letter-20261110-data-r1/output/audit"
samples=read(cpu/"probe_samples.json")["train_mixed"]
token_map=read(cpu/"tokenizer_a_ids.json")["token_to_id"]
index=read(ROOT/"vendor/LETTER/data/Instruments/Instruments.index.json")
readbacks={"r0":read(REVIEW/"gpu_r0_published_source_readback.json"),"r1":read(REVIEW/"gpu_published_source_readback.json")}
ledger=read(REVIEW/"gpu_failure_dependencies.json")["gpu_budget"]["value"]["runs"]
result={"status":"NEGATIVE_GPU_EVIDENCE_CONSISTENT","gpu_acceptance":"FAIL","jobs":[]}
for tag in ("r0","r1"):
    job_id="letter-20261110-gpu-"+tag
    d=ROOT/"evidence/jobs"/job_id
    manifest=read(d/"light-export-manifest.json")
    for entry in manifest["files"]:
        p=d/entry["path"];assert p.stat().st_size==entry["bytes"] and sha(p)==entry["sha256"],entry["path"]
    request,status,query=[read(d/name) for name in ("request.json","status.json","last-query.json")]
    assert status["status"]==query["status"]=="FAILED"
    assert status["exit_code"]==query["exit_code"]==1
    assert query["cleanup_verified"] and not query["worker_alive"] and not query["workgroup_live"]
    assert canon(request)==query["request_sha256"]
    inputs={pathlib.Path(x["path"]).name:x["sha256"] for x in request["inputs"]}
    for name,payload in readbacks[tag].items():
        raw=payload["structuredContent"]["content"].encode()
        assert hashlib.sha256(raw).hexdigest()==inputs[name],(tag,name)
    if tag=="r1":
        for name,expected in inputs.items():
            sub="config" if name.endswith(".json") else "scripts"
            assert sha(ROOT/sub/name)==expected,name
    rank_results=[]
    for rank in (0,1):
        a=read(d/"output/probe"/("rank"+str(rank)+".json"))
        assert a["status"]=="FAIL" and a["updates"]==[]
        assert a["source_commit"]==request["capture_version"]
        assert a["probe_identity"]["gpu_probe.py"]==inputs["gpu_probe.py"]
        assert a["probe_identity"]["compat.py"]==inputs["compat.py"]
        assert a["versions"]=={"torch":"2.6.0","transformers":"4.31.0","peft":"0.5.0","accelerate":"0.21.0","bitsandbytes":"0.45.5","deepspeed":"0.16.9","fschat":"0.2.31","flash-attn":"2.7.4.post1"}
        definition=a["training_definition"]
        assert definition["micro_batch"]==8 and definition["accumulation"]==4 and definition["global_batch"]==64
        assert a["deepspeed_config"]["zero_optimization"]["stage"]==2
        inv=a["parameter_inventory"]
        train={n for n,p in inv.items() if p["requires_grad"]}
        selected=[n for group in a["optimizer"]["groups"] for n in group["names"]]
        assert len(selected)==len(set(selected)) and set(selected)==train
        assert all(p["in_optimizer"]==(n in train) for n,p in inv.items())
        assert len(train)==450 and sum("lora_" in n for n in train)==448
        assert sum("modules_to_save.default" in n for n in train)==2
        assert sum(math.prod(inv[n]["shape"]) for n in train)==289652736
        assert all(not p["requires_grad"] and not p["in_optimizer"] for n,p in inv.items() if "original_module" in n)
        assert all(p["dtype_before_ds"]=="torch.int8" and not p["requires_grad"] for n,p in inv.items() if ".base_layer" in n)
        assert sum(p["dtype_before_ds"]=="torch.int8" for p in inv.values())==224
        diag=a["original_precision_diagnostic"]
        assert diag["status"]=="ORIGINAL_DTYPE_ERROR" and diag["matched_dtype_conflict"]
        assert diag["engine_global_steps_before"]==diag["engine_global_steps_after"]==0
        assert diag["optimizer_state_entries_before_after"]==[0,0]
        assert diag["rng_cpu_restored"] and diag["rng_cuda_restored"] and diag["sampled_parameters_unchanged"]
        first=a["mask_first_batch"];lengths=[]
        for i,selected_row in enumerate(samples[rank*8:rank*8+8]):
            ids,labels,attention=[first[k][i] for k in ("input_ids","labels","attention_mask")]
            length=sum(attention);lengths.append(length)
            assert length==selected_row["full_length_including_eos"]
            expected=[token_map[t] for t in index[selected_row["target_item_id"]]]+[2]
            assert [x for x in labels if x!=-100]==expected
            assert labels[:length-5]==[-100]*(length-5) and labels[length:]==[-100]*(len(labels)-length)
            assert ids[length-5:length]==expected and all(x==0 for x in ids[length:])
        p=a["lora_a_precision_adaptation"];assert len(p["targets"])==224
        combos=[v for t in p["targets"] for v in t["observed_dtype_combinations"]]
        assert all(v=={"input_dtype":"torch.float32","weight_dtype":"torch.bfloat16","output_dtype":"torch.bfloat16","outer_autocast":False,"linear_autocast":True,"autocast_after_call":False} for v in combos)
        if tag=="r0":
            assert a["error"]["message"]=="too many values to unpack (expected 4)"
            assert a["finite_checks"]["loss_microbatches_checked"]==0
        else:
            assert "MatMul8bitLtBackward" in a["error"]["message"] and "inplace" in a["error"]["message"]
            assert "checkpoint.py" in a["error"]["traceback"] and "result += output" in a["error"]["traceback"]
            assert a["finite_checks"]["loss_microbatches_checked"]==1
            loss=a["temperature_loss"]
            assert abs(loss["model"]-loss["independent_ce_temp08"])<1e-6
            assert abs(loss["model"]-loss["independent_ce_temp1"])>1e-3
            assert a["precision_context"]["observed_ce_logits_dtype"]=="torch.float32"
            assert not a["precision_context"]["observed_global_autocast_during_forward"] and not a["precision_context"]["observed_ce_global_autocast"]
            arity=a["fastchat_unpad_arity_compatibility"]
            assert arity["observed_arities"]==[5] and arity["extra_seqused_equals_mask_sum_and_cu_deltas"]
        assert "gradient_evidence" not in a and "adapter_save" not in a and "timing" not in a
        rank_results.append({"rank":rank,"completed_update_records":0,"finite_forward_microbatches":a["finite_checks"]["loss_microbatches_checked"],
             "first_batch_real_lengths":lengths,"cpu_ru_maxrss_gib":a["cpu_process_peak_rss_gib"],
             "gpu_peak_allocated_gib":a["gpu_final_peak_allocated_gib"],"gpu_peak_reserved_gib":a["gpu_final_peak_reserved_gib"],
             "temperature_loss":a.get("temperature_loss"),"error":a["error"]["message"]})
    resources=read(d/"output/resources/resource-summary.json")
    assert resources["stop_reason"] is None and resources["child_exit_code"]==1
    assert resources["charged_gpu_device_seconds"]==2*resources["elapsed_seconds"]
    assert resources==next(x for x in ledger if x["job_id"]==job_id)
    result["jobs"].append({"job_id":job_id,"source_commit":request["capture_version"],
        "manifest_files_verified":len(manifest["files"]),"published_source_bytes_matched":list(readbacks[tag]),
        "runner_status":"FAILED","exit_code":1,"cleanup_verified":True,"ranks":rank_results,
        "resources":{k:resources[k] for k in ("elapsed_seconds","charged_gpu_device_seconds","samples","peaks","stop_reason")}})
result["charged_gpu_device_seconds_total"]=sum(x["charged_gpu_device_seconds"] for x in ledger)
result["charged_gpu_device_minutes_total"]=result["charged_gpu_device_seconds_total"]/60
assert len(ledger)==2 and result["charged_gpu_device_seconds_total"]<7200
rounds=read(ROOT/"evidence/correction_rounds.json")
assert rounds["task_level_correction_round"]==rounds["max_correction_rounds"]==3
result["task_correction_rounds_used"]=3
result["fourth_correction_forbidden"]=True
result["96h_feasibility"]="UNKNOWN"
(REVIEW/"gpu_negative_artifact_verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":result["status"],"gpu_acceptance":"FAIL","jobs":len(result["jobs"]),"completed_optimizer_updates":0,
    "gpu_device_minutes":result["charged_gpu_device_minutes_total"],"rounds":3,"96h":"UNKNOWN"},indent=2))

