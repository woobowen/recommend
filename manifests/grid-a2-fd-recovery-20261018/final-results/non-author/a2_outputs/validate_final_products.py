from pathlib import Path
import csv,json
from numeric_core import ident,load
ROOT=Path(__file__).resolve().parent;CROSS=ROOT.parent/'cross_domain'
products=load(ROOT/'final-review-products.json')
for item in products['outputs']:
    actual=ident(item['path']);assert actual==item
base=load(CROSS/'cross-domain-evidence-index.json');final=load(CROSS/'cross-domain-evidence-index-with-a2.json')
assert ident(CROSS/'cross-domain-evidence-index.json')['sha256']=='c0b537e6700392165ceb5ff804d3d0e0070d3f2e6d63807fd91af72b7a1ee403'
assert base['pending_condition_count']==3 and final['pending_condition_count']==0 and final['available_condition_count']==24
assert final['overall_final_review_pass'] is False and len(final['routes'])==8
rows=load(CROSS/'cross-domain-original-metrics-with-a2.json')['rows'];assert len(rows)==24
old={r['route_id']:r for r in base['routes']}
for route in final['routes']:
    assert [x['shuffle_n'] for x in route['conditions']]==[0,3,4]
    for c in route['conditions']:
        row=next(x for x in rows if x['route']==route['route_id'] and x['condition']==c['condition'])
        for key,value in c['metrics_original'].items():assert row[key]==value
        if route['route_id']=='Sports-A2':
            source=load(ROOT/'terminal_outputs/raw/jobs'/c['job_id']/'output/RESULT.json')
            assert c['metrics_original']==source['metrics']['metrics'] and c['sid']['sha256']==source['sid']['sha256']
        else:
            prior=next(x for x in old[route['route_id']]['conditions'] if x['shuffle_n']==c['shuffle_n'])
            assert c==prior
with (CROSS/'cross-domain-original-metrics-with-a2.csv').open() as f:
    csvrows=list(csv.DictReader(f));assert len(csvrows)==24
    for c,j in zip(csvrows,rows):
        assert c['route']==j['route'] and c['condition']==j['condition']
        for key in ['recall@5','ndcg@5','recall@10','ndcg@10']:assert float(c[key])==j[key]
verified=set()
def refs(x):
    if isinstance(x,dict):
        if 'included_copy' in x:
            p=(CROSS/x['included_copy']).resolve();assert p.is_relative_to(CROSS)
            key=(str(p),x['sha256'])
            if key not in verified:
                a=ident(p);assert a['sha256']==x['sha256'] and a['bytes']==x['bytes'];verified.add(key)
        for value in x.values():refs(value)
    elif isinstance(x,list):
        for value in x:refs(value)
refs(final)
out={'status':'A2_OUTPUT_REVIEW_FINAL_PRODUCTS_VALIDATED','reviewer':'/root/toys_result_audit','source_condition_rows_exact':24,
     'old_non_A2_conditions_unchanged':21,'original_pending_index_unchanged':True,'CSV_binary_float_roundtrip_exact':True,
     'unique_included_references_hash_verified':len(verified),'products_manifest':ident(ROOT/'final-review-products.json'),
     'A2_raw_output_review':ident(ROOT/'A2_OUTPUTS_NONAUTHOR_REVIEW.json'),
     'A2_PNG_all_cells_review':ident(ROOT/'a2-png-pixel-consistency.json'),'A2_source_bindings_review':ident(ROOT/'a2-final-output-binding-checks.json'),
     'whole_task_FINAL_INTERNAL_REVIEW_not_signed_here':True,'GPU_model_or_evaluator_calls':0,'Git_or_remote_writes':0}
p=ROOT/'A2_REVIEW_DELIVERY_VALIDATION.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
