"""Make a dataset-parameterized copy of the verified LETTER-Beauty runner."""
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=HERE.parent/'letter_upstream_transformer_20261115/run_beauty.py'
old=source.read_text(encoding='utf-8')
new=old


def replace_once(before,after):
    global new
    assert new.count(before)==1,(before,new.count(before))
    new=new.replace(before,after,1)


replace_once('One fresh own-TIGER fit on official LETTER SIDs, then three inference modes.',
             'Own TIGER on official LETTER Instruments/Yelp SID tables, then three tests.')
replace_once("HERE=ROOT/'codex/letter_upstream_transformer_20261115'", "HERE=ROOT/'codex/letter_official_downstream_20261116'")
replace_once("    ap.add_argument('--output',type=Path,required=True)",
             "    ap.add_argument('--dataset',choices=['instruments','yelp'],required=True)\n    ap.add_argument('--output',type=Path,required=True)")
replace_once("audit_path=HERE/'inputs/beauty/mapping_audit.json'", "audit_path=HERE/'inputs'/args.dataset/'mapping_audit.json'")
replace_once("    assert audit['complete'] and audit['item_mapping_verified']", "    assert audit['complete'] and audit['item_mapping_verified'] and audit['dataset']==args.dataset")
replace_once("assert list(ids.shape)==[4,12101]", "assert list(ids.shape)==[4,audit['items']]")
replace_once("assert len(torch.unique(ids.T,dim=0))==12088", "assert len(torch.unique(ids.T,dim=0))==audit['unique_full_sids']")
replace_once("HERE/'prepare_beauty.py'", "HERE/'prepare_data.py'")
replace_once("dict(dataset='beauty',upstream='official_LETTER_released_SID'", "dict(dataset=args.dataset,upstream='official_LETTER_released_SID'")
replace_once("'++model.dataset_name=beauty'", "f'++model.dataset_name={args.dataset}'")
replace_once("old=ROOT/'experiments/sid_order_5seed_v1/beauty/shuffle/seed_42/.hydra/config.yaml'",
             "old=ROOT/'experiments/letter_official_beauty_20261115_125008/formal/.hydra/config.yaml'")
replace_once("result=dict(dataset='beauty',upstream='official_LETTER_SID'", "result=dict(dataset=args.dataset,upstream='official_LETTER_SID'")
replace_once("        rows=['| 模式 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |','|---|---:|---:|---:|---:|']",
             "        rows=['| 模式 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |','|---|---:|---:|---:|---:|']")
# Add explicit samples and checkpoint provenance in final output without changing model behavior.
replace_once("                    test_metrics={},completed=False)",
             "                    test_metrics={},completed=False,users=audit['users'],items=audit['items'],\n                    sid_sha256=audit['sid_sha256'],official_sid_collisions=audit['extra_colliding_items'])")
(HERE/'run_dataset.py').write_text(new,encoding='utf-8')
(HERE/'run_dataset.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
    fromfile='verified_beauty/run_beauty.py',tofile='official_instruments_yelp/run_dataset.py')),encoding='utf-8')
(HERE/'runner_source.json').write_text(json.dumps(dict(
    copied_from=str(source),source_sha256=hashlib.sha256(old.encode()).hexdigest(),
    generated_sha256=hashlib.sha256(new.encode()).hexdigest(),
    model_implementation_unchanged=True,train_seed=42,eval_seed=2026),indent=2)+'\n',encoding='utf-8')
compile(new,str(HERE/'run_dataset.py'),'exec')
print('Created dataset runner from verified Beauty pipeline; model and protocol unchanged.')
