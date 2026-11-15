#!/usr/bin/env python3
"""Derive a Beauty-scoped candidate from the exact historical A0 runner.

Candidate only: derivation is not deployment, authorization, or cleanup proof.
The public/shared runner is never modified. Review and CPU cleanup tests required.
"""
import argparse
import hashlib
import json
from pathlib import Path

HISTORICAL_SHA='fff2df040295bf9f342b8bb3144d05e28b5b2cc5aff41f8d47ae312847489bb5'
OLD="('letter-a0-20261111-','letter-a0-20261112-')"
NEW="('letter-a0-beauty-formal-','letter-a0-beauty-cpu-runner-check-')"

def derive(original):
 raw=original.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=HISTORICAL_SHA: raise ValueError('unexpected historical runner')
 text=raw.decode('utf-8')
 if text.count(OLD)!=2: raise ValueError('worker ownership anchors changed')
 candidate=text.replace(OLD,NEW)
 compile(candidate,str(original),'exec')
 return candidate.encode()

def main():
 p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.resolve()==a.original.resolve(): raise ValueError('original must remain unchanged')
 data=derive(a.original)
 with a.output.open('xb') as f:f.write(data)
 print(json.dumps({'status':'CANDIDATE_ONLY','historical_sha256':HISTORICAL_SHA,'candidate_sha256':hashlib.sha256(data).hexdigest(),'ownership_anchor_changes':2,'scope':'No deployment or execution authorization; namespace adaptation requires independent cleanup verification'}))
if __name__=='__main__':main()
