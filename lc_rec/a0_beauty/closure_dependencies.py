"""Import/path closure without importing Trainer or loading any model."""
import importlib,json,sys
from pathlib import Path
from formal_control import read,durable_write
from production_identity import need,identity,SYNTHETIC
from production_pipeline import actual_source_root
p=read(sys.argv[1]);need(p['evidence_scope']==SYNTHETIC,'CPU fixture source only')
release=Path(__file__).resolve().parents[2];version=read(release/'RUN_VERSION.json')
new=dict(p,release=str(release));source=actual_source_root(new)
need(actual_source_root(dict(new,source_root=str(source)))==source,'direct/wrapper author root mismatch')
sys.path.insert(0,str(release/'lc_rec/letter/scripts'))
imports={}
for name in ('formal_train','shuffle_runtime','shuffle_cpu_inputs','compat','formal_attention','formal_bnb_reload','trainer_loop_patch','zero2_partial_state','training_rows','epoch_schedule'):
 m=importlib.import_module(name);imports[name]=identity(m.__file__)
need(str(release) in imports['compat']['path'],'compat loaded from outside export')
import core
need(Path(core.__file__)==release/'lc_rec/shuffle/core.py','Shuffle core escaped fixed export')
from production_control import validate_plan
q=dict(new,run_commit=version['run_commit'],prediction_run_commit=version['run_commit'],
       config=str(release/'lc_rec/a0_beauty/config/formal_beauty_seed42.json'),inputs=version['files'])
validate_plan(q)
result={'scope':SYNTHETIC,'status':'PASS','no_model_loaded':True,'no_trainer_import':True,
        'normalized_author_source':str(source),'imports':imports,'core':identity(core.__file__),
        'export_commit':version['run_commit'],'config_author':identity(release/'lc_rec/letter/config/author_instruments.json')}
durable_write(Path(p['root'])/'dependency_check.json',result);print(json.dumps(result))
