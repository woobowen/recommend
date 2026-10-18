#!/usr/bin/env python3
"""Aggregate saved actual asset identities; never hash or deserialize remote assets."""
from pathlib import Path
import collections, datetime, hashlib, json, subprocess

OUT = Path(__file__).resolve().parent
SUPPORT = OUT.parents[1]
REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
OLD = Path('/home/addaswsw/recommend-support/grid-a-launch-20261017/deliveries/a2-emfile-capture-20261017T175451Z/package')
JOBS = '/home/bwwu/grid-audits/sid-extensions-p0/jobs/'
ENV = '/data/bwwu/envs/grid-20261009-e61baf3'
sources = {}

def source(key, path):
    p = Path(path)
    b = p.read_bytes()
    sources[key] = {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
    if p.is_relative_to(REPO):
        sources[key]['repository_path'] = str(p.relative_to(REPO))
        sources[key]['read_from_worktree_HEAD'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    return json.loads(b)

old = source('old_large_asset_manifest', OLD/'other-workflows/large-asset-manifest.json')
restore = source('original_selected_member_restoration', REPO/'manifests/grid-a-launch-20261017/reviews-and-identities/asset-restoration-receipt.json')
data = source('inherited_data_member_manifest', REPO/'manifests/grid-shuffle-20261017/asset-audit/inherited-data-sha-subset.json')
data_stat = source('previous_data_readonly_stat', REPO/'manifests/grid-shuffle-20261017/asset-audit/data-current-stat.json')
scene = source('final_scene', SUPPORT/'final-scene/scene-20261018T043428336611Z.json')
environment = source('historical_environment_metadata', REPO/'manifests/sid-extensions/ENVIRONMENT_METADATA.json')
env_observation = source('pre_recovery_CPU_environment_observation', OLD/'a2-code/dependency-and-source-identities/current-environment-observation.json')
install = source('original_installation_log', REPO/'manifests/grid-a-launch-20261017/INSTALLATION_LOG.json')
a1review = source('A1_independent_checkpoint_audit', SUPPORT/'reviews/sports_a1/remote_checkpoint_audit.json') if (SUPPORT/'reviews/sports_a1/remote_checkpoint_audit.json').exists() else None
a2review = source('A2_independent_terminal_training_audit', SUPPORT/'reviews/a2_training/terminal-review-20261018T042225Z.json')
restored = {a['path']: a for a in restore['assets']}
old_by_path = {a['path']: (i,a) for i,a in enumerate(old['assets'])}
assets = {}
routes = {r: {'checkpoint_asset_id': None, 'sid_asset_id': None} for r in ['Toys-A-5300','Toys-A-5000','Sports-A1','Sports-A2']}

def add(path, identity, asset_id, roles, refs, provenance, hash_basis):
    item = assets.get(path)
    if item is None:
        item = {'asset_id': asset_id, 'path': path, 'kind': 'regular_file', 'bytes': identity['bytes'], 'sha256': identity['sha256'], 'roles': [], 'source_receipts': [], 'provenance': provenance, 'hash_basis': hash_basis, 'payload_rehashed_by_this_inventory': False, 'read_only_recovery': {'host':'new-server','action':'Reuse this exact existing path read-only. Verify the recorded SHA before any future restore or use; do not overwrite historical producer jobs. No restore or experiment is executed by this inventory.'}}
        assets[path] = item
    assert item['bytes'] == identity['bytes'] and item['sha256'] == identity['sha256'], path
    item['roles'] = sorted(set(item['roles'] + roles))
    item['source_receipts'].extend(refs)
    if path in restored:
        a = restored[path]
        item['original_archive_recovery'] = {k:a[k] for k in ['capture_version','batch_id','archive_path','manifest_sha256']}
        item['read_only_recovery']['selected_member_restore_entrance'] = 'Original preserved archive capture/member and original_selected_member_restoration receipt; restore a new independent entity only after separately authorized storage recovery, then verify member SHA.'
    return item['asset_id']

for route, tag in [('Toys-A-5300','toys5300'), ('Toys-A-5000','toys5000')]:
    p = SUPPORT/f'reviews/toys/raw/ga-20261017-{tag}-eval-0-v1/output/loading.json'
    key = tag+'_actual_loading'
    load = source(key,p)
    for i,a in enumerate(load['assets']):
        role = 'sid' if i == 0 else 'checkpoint'
        asset_id = ('toys_shared_sid' if role=='sid' else tag+'_checkpoint')
        ident = add(a['path'],a,asset_id,[route+'/'+role],[{'source':key,'json_pointer':f'/assets/{i}'}], 'A800 preserved original selected member; historical upstream SID producer remains non-unique.', 'SHA recorded by real successful job loading and original archive restoration; no new content hash here.')
        routes[route][role+'_asset_id'] = ident

for path,(i,a) in old_by_path.items():
    if ('ga-20261017-sportsa1-train-v1/output/' in path and path.endswith('.ckpt')) or ('sports_rkmeans_qwen_' in path and path.endswith('/semantic_ids.pt')):
        role = 'sid' if path.endswith('/semantic_ids.pt') else ('checkpoint' if '/checkpoints/' in path else 'final_full_state')
        ident = add(path,a,'sports_a1_'+role,['Sports-A1/'+role],[{'source':'old_large_asset_manifest','json_pointer':f'/assets/{i}'}], 'A1 newly trained dual-table generator with actual Qwen SID; original A1 execution source preserved separately.', a['identity_basis'])
        if role in ['sid','checkpoint']: routes['Sports-A1'][role+'_asset_id'] = ident

for job, selected in [
    ('ga-20261017-sportsa2-encode-v1', ['output/item_embeddings.pt','output/predictions.pt','output/prediction-rank-0.pt']),
    ('ga-20261018-sportsa2-quantize-fd-recovery-v1', ['output/quantizer.ckpt','output/checkpoints/quantizer_000030.ckpt','output/semantic_ids.pt','output/predictions.pt']),
    ('ga-20261018-sportsa2-train-fd-recovery-v1', ['output/checkpoints/checkpoint_epoch=000_step=006900.ckpt','output/recovery/training-final/full-state.ckpt']),
]:
    query = scene['jobs'][job]['query']
    assert query['status']=='SUCCEEDED' and query['exit_code']==0 and query['cleanup_verified'] is True
    found = set()
    for i,a in enumerate(query['artifacts']):
        if a['path'] not in selected: continue
        found.add(a['path'])
        path=JOBS+job+'/'+a['path']
        if 'encode-v1' in job:
            role={'output/item_embeddings.pt':'reused_input_embedding','output/predictions.pt':'encoder_prediction_tensor','output/prediction-rank-0.pt':'encoder_prediction_rank0'}[a['path']]
        elif 'quantize' in job:
            role={'output/quantizer.ckpt':'quantizer_final','output/checkpoints/quantizer_000030.ckpt':'quantizer_step30','output/semantic_ids.pt':'sid','output/predictions.pt':'raw_quantized_three_codes'}[a['path']]
        else: role='checkpoint' if '/checkpoints/' in a['path'] else 'final_full_state'
        ident=add(path,a,'sports_a2_'+role,['Sports-A2/'+role],[{'source':'final_scene','json_pointer':f'/jobs/{job}/query/artifacts/{i}'}], 'Original successful T5 encoder reused unchanged.' if 'encode-v1' in job else 'New A2 FD recovery actual successful immutable job; source85b77f5e2a5b497fdc1f620649ba488fc783a245 snapshot22286966cd544773.', 'Saved runner terminal artifact SHA; read from final live scene. Encoder hash inherited original success; new A2 quantizer/training hashes generated by terminal runner and independently checked in current recovery audits.')
        if role in ['checkpoint','sid']:routes['Sports-A2'][role+'_asset_id']=ident
    assert found==set(selected), (job,found,set(selected))

for i,a in enumerate(restore['assets']):
    if '/A800_T5_SMALL/' not in a['path']:continue
    add(a['path'],a,'t5_small_'+Path(a['path']).name,['Sports-A2/reused_T5_small'],[{'source':'original_selected_member_restoration','json_pointer':f'/assets/{i}'}], 'A800-project-models-20261010-v1 original selected member; frozen T5 encoder inputs reused.', 'Original restored member content SHA verified then; large weight payload not rehashed by this inventory.')

datasets = []
for i,wrapper in enumerate(data['records']):
    a=wrapper['original_record']
    datasets.append({'path':a['actual_server_path'],'bytes':a['bytes'],'sha256':a['sha256'],'dataset':a['dataset'],'split':a['split'],'kind':'regular_file','hash_basis':'Inherited original member SHA; no current payload hashing or parsing in this inventory.','payload_rehashed_by_this_inventory':False,'source_receipt':{'source':'inherited_data_member_manifest','json_pointer':f'/records/{i}/original_record','original_manifest_physical_line_1based':wrapper['source_physical_line_1based']},'original_archive_recovery':{k:a[k] for k in ['source_version','source_original_path','matching_preserved_capture','archive_batch','archive_member','entry_id']},'read_only_recovery':a['recovery']})
assert len(datasets)==1293
directories=[]
for (dataset,split),members in __import__('itertools').groupby(sorted(datasets,key=lambda a:(a['dataset'],a['split'])),lambda a:(a['dataset'],a['split'])):
    rows=list(members)
    directories.append({'kind':'directory','path':str(Path(rows[0]['path']).parent),'dataset':dataset,'split':split,'member_count':len(rows),'logical_member_bytes':sum(a['bytes'] for a in rows),'sha256':None,'sha256_scope':'No single directory payload SHA exists. Members are individually hashed; this is not a filesystem image.','member_list_source':'dataset_members in this manifest','original_member_manifest':data['source_manifest'],'member_manifest_receipt_source':'inherited_data_member_manifest','payload_rehashed_by_this_inventory':False})
t5_rows=[a for a in assets.values() if '/A800_T5_SMALL/' in a['path']]
directories.append({'kind':'directory','path':str(Path(t5_rows[0]['path']).parent),'member_count':len(t5_rows),'logical_member_bytes':sum(a['bytes'] for a in t5_rows),'sha256':None,'sha256_scope':'No single directory SHA; six preserved restored members individually identified in assets.','member_list_source':'assets with Sports-A2/reused_T5_small role','member_manifest_receipt_source':'original_selected_member_restoration'})
directories.append({'kind':'directory','path':ENV,'member_count':None,'logical_member_bytes':None,'sha256':None,'sha256_scope':'Environment tree was not recursively scanned, sized, copied, or content hashed. Version manifest is not a bit-for-bit environment archive.','member_manifest_receipt_source':'historical_environment_metadata','recovery_entrance':ENV+'/bin/python','gap':'No complete per-file environment payload manifest. Historical package metadata paths under python3.1 are preserved verbatim; their path correctness is not asserted by this inventory.'})
assert all(v['checkpoint_asset_id'] and v['sid_asset_id'] for v in routes.values())
assert len({a['path'] for a in datasets})==len(datasets)
assert len({a['path'] for a in assets.values()})==len(assets)
manifest={'status':'SAVED_IDENTITIES_AGGREGATED_PENDING_TARGETED_STAT','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'/root/sports_a1_result_audit','scope':'Four A routes checkpoint/SID, new A2 quantizer, original reused T5 outputs and weights, all six Sports/Toys TFRecord splits, environment version identity. No new experiment or restore.','routes':routes,'assets':list(assets.values()),'directories':directories,'dataset_members':datasets,'source_receipts':sources,'hash_observation_policy':{'historical_hashes_are_not_newly_measured':True,'remote_big_payload_rehashes_this_inventory':0,'GPU_or_model_calls':0,'server_recursive_scans':0,'large_asset_downloads':0},'unverified_or_limited':['Original Toys upstream SID producer remains non-unique.','New A2 is method reproduction; no recovered historical Sports generator weights.','A directory has no single content SHA: use individual member identities and referenced manifest SHA.','Environment metadata records package versions, not a complete per-file executable/runtime image.','Saved SHA inheritance plus fresh size/stat does not prove fresh current payload byte equality; current independent model/SID audits are separate receipts.']}
(OUT/'asset-manifest.pending-stat.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(OUT/'environment-manifest.json').write_text(json.dumps({'historical_environment_metadata':environment,'current_pre_recovery_probe':env_observation,'installation_log':install,'source_receipts':{k:sources[k] for k in ['historical_environment_metadata','pre_recovery_CPU_environment_observation','original_installation_log']},'directory_sha256':None,'per_file_environment_manifest_available':False},ensure_ascii=False,indent=2)+'\n')
paths=sorted(set([a['path'] for a in assets.values()]+[a['path'] for a in datasets]+[a['path'] for a in directories]+[ENV+'/bin/python',ENV+'/bin/python3.10','/home/bwwu/recommend/installed-requirements.txt']))
(OUT/'read-only-probe-request.json').write_text(json.dumps({'paths':paths,'directories':[a['path'] for a in directories],'hash_small_files_only':['/home/bwwu/recommend/installed-requirements.txt',ENV+'/bin/python3.10'],'per_file_hash_upper_bytes':33554432,'interpreter':ENV+'/bin/python','current_version_probe_packages':['torch','lightning','pytorch-lightning','tensorflow-cpu','hydra-core','omegaconf','numpy','fsspec','transformers']},indent=2)+'\n')
print(json.dumps({'assets':len(assets),'dataset_members':len(datasets),'dataset_logical_bytes':sum(a['bytes'] for a in datasets),'directories':len(directories),'paths_for_targeted_stat':len(paths),'routes':routes},ensure_ascii=False))
