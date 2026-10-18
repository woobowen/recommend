"""Local bounded CPU probes; no claim about still-future A2 output results."""
from pathlib import Path
import ast,copy,json
import numpy as np
from numeric_core import ident,load,read_bundle,independent_paths,ranking
ROOT=Path(__file__).resolve().parent
prep=load(ROOT/'preparation-verification.json');assert prep['not_final_scientific_result_acceptance']
for path in ROOT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
# Existing non-A2 raw artifact is an identified tooling fixture, never relabeled as A2.
fixture=ROOT.parent/'toys/raw/ga-20261017-toys5300-attention-0-v1/output/batch-00/bundle/record.json'
b=read_bundle(fixture);paths,err=independent_paths(b['steps'])
for k,v in paths.items():assert np.array_equal(v,b['paths'][k])
damaged=copy.deepcopy(b['steps']);damaged[1]['parent_indices'][0,0]=(damaged[1]['parent_indices'][0,0]+1)%10
rejected=False
try:independent_paths(damaged)
except AssertionError:rejected=True
assert rejected
ambiguous={'sids':[[i,0,0,0] for i in range(10)],'scores':[10,9,8,7,6,6,4,3,2,1],'target':[5,0,0,0]}
rejected_tie=False
try:ranking(ambiguous)
except AssertionError:rejected_tie=True
assert rejected_tie
out={'status':'A2_OUTPUT_AUDIT_TOOLING_READY_RESULTS_PENDING','reviewer':'/root/toys_result_audit',
     'preparation_verification':ident(ROOT/'preparation-verification.json'),
     'syntax_parse_all_scripts':True,'independent_path_fixture':ident(fixture),'fixture_is_A2':False,
     'all_fixture_paths_match':True,'altered_parent_mapping_rejected':True,'relevant_score_tie_rejected':True,
     'numeric_fixture_tolerance_not_scientific_similarity_threshold':True,
     'scripts':[ident(ROOT/name) for name in ['collect_evidence.py','numeric_core.py','verify_preparation.py','audit_terminal_outputs.py','check_tooling_readiness.py']],
     'terminal_A2_output_review_run':False,'overall_final_internal_review_pass':False,
     'await_main_controller_trigger':True,'GPU_calls':0,'models_loaded':0,'server_mutations':0,
     'pending_expected_jobs':[f'ga-20261018-sportsa2-{role}-fd-recovery-v1' for role in ['eval-0','eval-3','eval-4','attention-0','attention-3','attention-4','stats']]}
p=ROOT/'tooling-readiness.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'pending_jobs':7,'A2_output_PASS':False,'readiness':ident(p)}))
