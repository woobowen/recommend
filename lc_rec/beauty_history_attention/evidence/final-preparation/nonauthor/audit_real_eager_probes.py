"""Review exact real native-eager probe artifacts without loading any model."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args=parser.parse_args()
def read(path):
    return json.loads(path.read_text())
old_folder=args.root/"old-causal-r0"
b_folder=args.root/"b-mask-r0"
old=read(old_folder/"result.json")
assert old["status"]=="PASS" and old["kind"]=="old-causal"
assert old["row_ids"]==[0,8,734]
assert old["adapter_files"]["adapter_model.bin"]["sha256"]=="dd9f3d2983af7ce85ace99534bd23df9362d9cd91ac39d0b356b219cb790b1c1"
observed=old["actual_checks"]
assert observed["same_weight_same_native_path_exact"]
assert observed["attention"]["backend"]=="native_eager"
assert observed["attention"]["cache_reorders"]==15
case_rows=[]
for row_idx,case in zip(old["row_ids"],observed["cases"],strict=True):
    adapted=[{key:c[key] for key in ("generated_token_ids","sid","sequence_score")} for c in case["adapted"]]
    original=case["original"]
    assert adapted==original and len(original)==20
    assert case["logits"]["max_abs"]==case["logits"]["rms"]==0.0 and case["logits"]["same_argmax"]
    order=sorted(range(20),key=lambda j: (-original[j]["sequence_score"],j))
    assert case["stable_order"]==order
    boundary=case["actual_boundary"]
    assert boundary["beam_expansion"]==20 and boundary["cache_length"]==0
    assert len(boundary["decodes"])==4 and boundary["position_ids"]==list(range(len(boundary["input_ids"])))
    case_rows.append({"row_idx": row_idx,"n_beams":20,"native_scores_and_order_exact":True,"logits_max_abs":0.0})
result={"same_weight_original_A":case_rows,"old_A_native_source_sha256":observed["attention"]["native_source_sha256"],
        "old_A_launch":read(old_folder/"execution.json"),"old_A_cuda_peak_reserved_bytes":old["peak_reserved_bytes"]}
b=read(b_folder/"result.json")
assert b["status"]=="PASS" and b["kind"]=="b-mask" and b["row_ids"]==[8]
assert b["adapter_identity"]["checkpoint_purpose"]=="preflight"
assert b["adapter_identity"]["adapter_sha256"]=="a9ce5337e30d0ffe6917755455c18f6544ad9a73c4a29cf6e656e6872ccb3665"
x=b["actual_checks"]
assert x["actual_layer_weights"]["layers"] and len(x["actual_layer_weights"]["layers"])==32
assert x["all_layer_history_states_exact"] and x["earlier_logits_exact"]
assert len(x["actual_beam_candidates"])==20 and x["beam_boundary"]["beam_expansion"]==20
assert len(x["beam_boundary"]["decodes"])==4 and x["attention"]["cache_reorders"]==5
h=x["beam_boundary"]["history_sid_mask"]
number_history=sum(h)
true_len=len(h)+5
physical_len=4+true_len+3
allowed_graph_edges=true_len*(true_len+1)//2 + number_history*(number_history-1)//2
reported=x["actual_layer_weights"]
assert reported["allowed_edge_count"]==allowed_graph_edges
assert reported["added_edges"]==number_history*(number_history-1)//2
for i,event in enumerate(reported["layers"]):
    assert event["layer"]==i and event["shape"]==[1,32,physical_len,physical_len]
    assert event["dtype"]=="torch.bfloat16"
    assert event["positive_reverse_history_weights"] > 0
    assert 0 < event["cross_item_heads_with_positive_weight"] <= 32
cache=x["cache_logits"]
ratio=cache["rms"]/cache["ref_rms"]
assert ratio < .03 and cache["same_argmax"]
assert set(x["eager_vs_explicit_fp32_contexts"])=={"0","31"}
reference={k:v["rms"]/v["ref_rms"] for k,v in x["eager_vs_explicit_fp32_contexts"].items()}
assert all(v<.015 for v in reference.values())
assert set(x["operators"]) >= {"aten::_softmax","aten::bmm"}
assert all("scaled_dot_product" not in item and "efficient_attention" not in item for item in x["operators"])
result["bidirectional_actual"]={"row_idx":8,"history_tokens":number_history,"added_edges":reported["added_edges"],
                                  "layers_checked":32,"position_and_cache_reorder_checks":True,
                                  "cache_abs":cache["max_abs"],"cache_relative_rms":ratio,"reference_relative_rms":reference,
                                  "operators":x["operators"],"launch":read(b_folder/"execution.json"),
                                  "peak_reserved_bytes":b["peak_reserved_bytes"] }
print(json.dumps(result,indent=2,allow_nan=False))
