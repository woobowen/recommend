"""CPU-only original-evaluator recomputation of the bounded saved preflight."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("plan",type=Path)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    snapshot=Path(plan["snapshot"]);sys.path.insert(0,str(snapshot))
    from experiments.beauty_extensions.common import bind_source, load_config, verify_asset, write_json
    from experiments.beauty_extensions.results import recompute
    output=Path(os.environ["GRID_JOB_DIR"])/"output"
    rows=[]
    for item in plan["entries"]:
        spec=item["spec"];bind_source(spec["route"])
        folder=output/item["name"];folder.mkdir()
        cfg=load_config(spec,folder)
        verify_asset(item["predictions"]);verify_asset(item["metrics"])
        rows.append({"name":item["name"],**recompute(spec,cfg,item["predictions"]["path"],folder,
                    trainer_metrics=item["metrics"]["path"])})
    write_json(output/"RESULT.json",{"status":"PASS","engineering_only":True,"rows":rows})


if __name__=="__main__":main()
