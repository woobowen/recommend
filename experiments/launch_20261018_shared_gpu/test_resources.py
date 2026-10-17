"""Meaningful CPU checks of memory/load sharing and frozen GPU allocation."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.launch_20261017 import queue as q

G0='GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae'
G1='GPU-59cac3bd-3645-668b-79c8-839e0d878893'

def stage(kind='evaluate',devices=None):
    peak,margin=(20480,2048) if kind=='train' else ((6144,2048) if kind=='rkmeans' else (2048,1024))
    return {'kind':kind,'gpu_uuids':devices or [G0], 'resource_policy':{
        'allow_shared':True,'estimated_peak_mib':peak,'safety_margin_mib':margin,
        'min_runtime_free_mib':512,'max_prelaunch_utilization_percent':50,
        'required_samples':3,'sample_seconds':2}}

def card(free=15000,used=9000,load=30,processes=None):
    return {'total_mib':24564,'used_mib':used,'free_mib':free,'utilization':load,
        'processes':processes if processes is not None else [{'pid':999,'used_mib':used,'process_name':'foreign'}]}

class ResourceTests(unittest.TestCase):
    def test_peer_six_gib_is_not_a_veto(self):
        cards={G0:card(17564,7000)}
        self.assertTrue(q.resource_ok(stage(),cards))
        self.assertTrue(q.resource_ok(stage('rkmeans'),cards))
        self.assertFalse(q.resource_ok(stage('train'),cards))

    def test_real_free_accounts_for_driver_reserved_memory(self):
        # total-used would pass 22528; actual free does not.
        self.assertFalse(q.resource_ok(stage('train'),{G0:card(22200,1800)}))

    def test_both_quantizer_cards_must_fit(self):
        s=stage('rkmeans',[G0,G1]);cards={G0:card(22000,2000),G1:card(7000,17000)}
        self.assertFalse(q.resource_ok(s,cards))
        self.assertTrue(q.resource_ok(stage(),{G0:cards[G0]}))

    def test_sustained_prelaunch_load_limit(self):
        self.assertFalse(q.resource_ok(stage(),{G0:card(load=51)}))
        self.assertTrue(q.resource_ok(stage(),{G0:card(load=50)}))

    def test_our_two_ddp_ranks_are_not_peers_and_own_memory_is_deducted(self):
        s=stage('train');cards={G0:card(free=3400,used=20684,load=100,processes=[
            {'pid':1,'used_mib':10000},{'pid':2,'used_mib':10160},{'pid':3,'used_mib':524}])}
        with patch.object(q.os,'getpgrp',return_value=7),patch.object(q.os,'getpgid',side_effect=lambda pid:7 if pid in (1,2) else 8):
            a=q.resource_assessment(s,cards,True)[G0]
            self.assertTrue(a['ok']);self.assertEqual(a['owned_mib'],20160)
            self.assertEqual(a['required_free_mib'],2368)
            cards[G0]['free_mib']=1000
            self.assertFalse(q.resource_ok(s,cards,True))

    def test_sampled_memory_growth_adds_headroom(self):
        s=stage('rkmeans')
        samples=[{G0:card(free=9000,used=15000)},{G0:card(free=10000,used=14000)},{G0:card(free=9000,used=15000)}]
        with patch.object(q,'gpu_snapshot',side_effect=samples),patch.object(q.time,'sleep'):
            okay,records=q.prelaunch_samples(s)
        self.assertFalse(okay)
        self.assertEqual(records[-1]['resource_assessment'][G0]['sampled_required_free_mib'],9192)

    def test_alternative_allocation_is_frozen_without_changing_research_request(self):
        s=stage();s['gpu_candidates']=[G0,G1];s['request']={'argv':['/python','original.py'],'env':{'CUDA_VISIBLE_DEVICES':G0}}
        selected=q.allocation_stage(s,[G1])
        self.assertEqual(q.gpus(selected),[G1]);self.assertEqual(selected['request'],s['request'])
        self.assertEqual(q.gpus(s),[G0])
        with self.assertRaises(ValueError):q.allocation_stage(s,['GPU-0000'])

if __name__=='__main__':unittest.main()
