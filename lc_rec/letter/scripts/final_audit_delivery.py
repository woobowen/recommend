#!/usr/bin/env python3
"""Narrow final CPU audit package; inherit verified ZIP and Windows checks."""
import formal_delivery


def selected(name):
    base='lc_rec/letter/'
    r=name.removeprefix(base)
    if r.startswith(('evidence/final_audit_20261113/', 'review/final_audit_20261113/', 'config/')):
        return True
    if r in ('A0_NORMAL_FINAL_REVIEW_ENTRY.md','EVALUATION_PROTOCOL_V1.md',
             'review/formal_det_20261112/independent_metrics.py',
             'review/formal_det_20261112/prelaunch_training.md',
             'review/formal_det_20261112/prelaunch_pipeline.md',
             'evidence/formal_20261111/precision_approval.json',
             'evidence/formal_det_20261112/model_identity.json',
             'evidence/formal_det_20261112/environment.json',
             'evidence/formal_det_20261112/run_github_readback.json',
             'evidence/formal_det_20261112/deployments/4c6a83d9ba42dc2fe737792759e410ef1874da76.json'):
        return True
    if r.startswith('evidence/formal_det_20261112/observations/letter-a0-20261112-normal-det-seed42-r0/status_check_20261113T082550Z/'):
        return True
    if r.startswith('scripts/') and name.endswith('.py'):
        return True
    if r.startswith('vendor/LETTER/LETTER-LC-Rec/') and name.endswith(('.py','.sh')):
        return True
    return False


if __name__=='__main__':
    formal_delivery.selected=selected
    formal_delivery.main()
