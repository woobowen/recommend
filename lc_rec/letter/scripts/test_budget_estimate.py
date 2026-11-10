"""Pure-stdlib budget arithmetic regressions; synthetic timings are not GPU evidence."""
import importlib.util
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('letter_estimate_budget', SCRIPTS / 'estimate_budget.py')
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)
spec2 = importlib.util.spec_from_file_location('letter_inference_probe', SCRIPTS / 'inference_probe.py')
inference = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(inference)


def fixture():
    audit = {'status': 'PASS', 'budget_counts': {'train_rows_per_epoch': 131837,
        'valid_two_prompt_rows_per_epoch': 49544, 'test_prompt0_rows': 24772, 'epochs': 4, 'global_batch': 64},
        'original_dataset_execution': {'train_rows': 131837, 'valid_rows_after_two_prompts': 49544, 'test_rows': 24772},
        'length_distributions': {'train': {'all_prompt_full_with_eos': {'count': 2241229, 'min': 62,
           'max': 175, 'mean': 92.5, 'quantiles': {'0.5': 83}}},
            'test': {'input_per_prompt': [{'histogram': {'70': 10000, '80': 10000, '100': 4772}}]}}}
    training, generating = [], []
    for rank in range(2):
        updates = []
        for number, phase, seconds in [(0, 'warmup', 100), (1, 'warmup', 100), (2, 'warmup', 100),
                (3, 'mixed_steady', 10), (4, 'mixed_steady', 10), (5, 'mixed_steady', 10),
                (6, 'short', 5), (7, 'middle', 10), (8, 'long', 20)]:
            updates.append({'update': number, 'phase': phase, 'max_rank_seconds': seconds,
                'global_samples': 64, 'effective_input_tokens': 6000, 'padded_tokens': 10000,
                'cpu_batch_preparation_and_mask_verification_seconds': 1,
                'rank_true_lengths': [90] * 32})
        training.append({'rank': rank, 'status': 'PASS', 'updates': updates,
            'training_definition': {'formal_dataloader_batches_per_rank': 8240, 'formal_scheduler_updates': 8240},
            'validation_timings': [{'phase': phase, 'max_rank_seconds': seconds, 'global_samples': 16,
                                   'cpu_batch_preparation_and_mask_verification_seconds': .5,
                                   'rank_lengths': [100] * 8} for phase, seconds in [('warmup', 100), ('mixed', 1), ('long', 2)]],
            'adapter_save': {'write_seconds_rank0': 2, 'seconds_rank0': 3},
            'stages': [{'stage': 'before_updates', 'elapsed_seconds': 10}]})
        generating.append({'rank': rank, 'status': 'RELOAD_BEAM20_TECHNICAL_PASS', 'base_load_seconds': 5,
            'adapter_reload_seconds': 1,
            'timings': [{'warmup': warmup, 'selection': phase, 'row_idx': number * 2 + rank,
                'input_length': length, 'max_rank_seconds': seconds, 'global_users': 2,
                'cpu_collator_and_transfer_seconds': .1}
                for number, (phase, length, seconds, warmup) in enumerate([
                    ('q10', 70, 100, True), ('q10', 70, 2, False), ('q50', 80, 3, False),
                    ('q90', 100, 4, False), ('max', 159, 5, False)])]})
    return audit, training, generating


class BudgetTests(unittest.TestCase):
    def test_exact_aggregate_and_warmup_exclusion(self):
        a, train, generation = fixture()
        result = budget.estimate(a, train, generation, 0, 0)
        self.assertEqual(result['observed_counts']['trainer431_floor_scheduler_updates_four_epochs'], 8240)
        self.assertEqual(result['observed_counts']['conservative_optimizer_updates_four_epochs'], 8240)
        self.assertEqual(result['measurements']['steady_mixed_update_count'], 3)
        self.assertEqual(result['measurements']['steady_mixed_global_samples_per_second'], 6.4)
        self.assertEqual(result['measurements']['steady_mixed_effective_tokens_per_second'], 600)
        self.assertAlmostEqual(result['center_estimate']['component_hours']['train'], 8240 * 11 / 3600)
        self.assertAlmostEqual(result['conservative_estimate']['component_hours']['train'], 8240 * 21 / 3600)
        self.assertAlmostEqual(result['center_estimate']['component_hours']['validation'], 49544 * 4 * 1.5 / 16 / 3600)
        self.assertAlmostEqual(result['conservative_estimate']['component_hours']['validation'], 49544 * 4 * 2.5 / 16 / 3600)
        beam_seconds = (10000 * 2.1 + 10000 * 3.1 + 4772 * 4.1) / 2
        self.assertAlmostEqual(result['center_estimate']['component_hours']['beam20_prompt0_test'], beam_seconds / 3600)
        self.assertFalse(result['center_estimate']['is_full_seed_cost_validation'])
        self.assertIn('INCONCLUSIVE', result['time_limit_assessment'])

    def test_failed_real_chain_never_produces_rate(self):
        a, train, generation = fixture()
        for rank in train:
            rank['status'], rank['updates'] = 'FAIL', []
        result = budget.estimate(a, train, [])
        self.assertEqual(result['status'], 'INCONCLUSIVE')
        self.assertIsNone(result['center_estimate']['total_hours'])
        self.assertIsNone(result['measurements']['steady_mixed_global_samples_per_second'])

    def test_missing_generation_is_not_complete_estimate(self):
        a, train, generation = fixture()
        result = budget.estimate(a, train, [])
        self.assertIn('beam20_prompt0_test', result['center_estimate']['missing_components'])
        self.assertIsNone(result['center_estimate']['total_hours'])

    def test_duplicate_rank_users_rejected(self):
        a, train, generation = fixture()
        generation[1]['timings'][1]['row_idx'] = generation[0]['timings'][1]['row_idx']
        with self.assertRaisesRegex(AssertionError, 'Duplicate users'):
            budget.estimate(a, train, generation)

    def test_nonfinite_rate_rejected(self):
        a, train, generation = fixture()
        train[0]['updates'][3]['max_rank_seconds'] = float('nan')
        with self.assertRaisesRegex(AssertionError, 'finite positive'):
            budget.estimate(a, train, generation)

    def test_inventory_mapping_keeps_sid_and_lora_names(self):
        self.assertEqual(inference.saved_key('base.model.q_proj.lora_A.default.weight'), 'base.model.q_proj.lora_A.weight')
        self.assertEqual(inference.saved_key('base.model.embed_tokens.modules_to_save.default.weight'), 'base.model.embed_tokens.weight')
        with self.assertRaisesRegex(AssertionError, 'Unexpected trained tensor'):
            inference.saved_key('base.model.embed_tokens.original_module.weight')


if __name__ == '__main__':
    unittest.main(verbosity=2)
