#!/usr/bin/env python3
"""Standard-library-only tests against the actual fixed author evaluator."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import formal_evaluate as entry


SOURCE = Path(__file__).resolve().parents[1] / "vendor/LETTER/LETTER-LC-Rec"


class FormalEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.author = entry.load_evaluator(SOURCE)
        self.catalog = {"A": ["10", "11"], "B": ["12"], "C": ["13"]}

    def prediction(self, index, target, sids, scores=None, rank=0):
        self.assertEqual(len(sids), 20)
        scores = [20 - i for i in range(20)] if scores is None else scores
        order = sorted(range(20), key=lambda i: scores[i], reverse=True)
        user = {"row_idx": index, "user_id": f"user{index}", "target_item_id": "10" if target == "A" else "13", "target": target}
        row = {**user, "rank": rank, "prompt_id": 0,
               "candidates": [{"generation_order": i, "raw_decoded": "instruction\n### Response: " + sid + " \n",
                               "sequence_score": scores[i], "sid": sid}
                              for i, sid in enumerate(sids)],
               "sorted_generation_order": order,
               "relevance": [int(sids[i] == target) for i in order]}
        return row, user

    def test_original_duplicate_ndcg_and_illegal_sid_are_retained(self):
        row, user = self.prediction(0, "A", ["INVALID", "A", "A"] + ["B"] * 17)
        metrics, counts = entry.audit_prediction(row, user, self.catalog, self.author)
        self.assertEqual(metrics["hit@5"], 1)
        self.assertAlmostEqual(metrics["ndcg@5"], 1 / math.log2(3) + 1 / math.log2(4))
        self.assertGreater(metrics["ndcg@5"], 1)
        self.assertEqual(counts["invalid_complete_sid_candidates"], 1)
        self.assertEqual(counts["duplicate_candidate_occurrences"], 17)
        self.assertEqual(counts["colliding_sid_candidate_occurrences"], 2)
        self.assertEqual(counts["targets_with_colliding_sid"], 1)

    def test_illegal_beams_are_not_filtered_and_ties_keep_generation_order(self):
        row, user = self.prediction(0, "A", ["INVALID"] * 5 + ["A"] + ["B"] * 14, scores=[0] * 20)
        metrics, _ = entry.audit_prediction(row, user, self.catalog, self.author)
        self.assertEqual(metrics["hit@5"], 0)
        self.assertEqual(metrics["hit@10"], 1)
        self.assertEqual(row["sorted_generation_order"], list(range(20)))
        self.assertAlmostEqual(metrics["ndcg@10"], 1 / math.log2(7))

    def test_original_parsing_exactly_matches_spaces_and_last_response(self):
        self.assertEqual(entry.normalized_prediction("Response: wrong Response:  <a_1> <b_2> \n"), "<a_1><b_2>")
        self.assertEqual(entry.normalized_prediction("Response: <a_1>\t<b_2>"), "<a_1>\t<b_2>")

    def test_full_key_inventory(self):
        shapes = entry.expected_adapter_shapes()
        self.assertEqual(len(shapes), 450)
        self.assertEqual(sum(".lora_" in key for key in shapes), 448)
        self.assertEqual(sum(a * b for a, b in shapes.values()), 289652736)

    def test_aggregate_exact_coverage_and_final_rank_status(self):
        row0, user0 = self.prediction(0, "A", ["A"] + ["B"] * 19)
        row1, user1 = self.prediction(1, "C", ["B"] * 20, rank=1)
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f"predictions.rank{rank}.jsonl" for rank in range(2)]
            for rank, (row, path) in enumerate(zip((row0, row1), paths)):
                path.write_text(json.dumps(row) + "\n")
                (Path(folder) / f"rank{rank}.json").write_text(json.dumps({
                    "rank": rank, "RUN_COMMIT": "a" * 40, "status": "PASS", "completed_users": 1,
                    "adapter_files": {}, "data_identity": {}, "author_sources": {}, "generation": {},
                    "allowed_callback_sha256": "example", "elapsed_generation_seconds": 1.0}))
            result = entry.aggregate_predictions(paths, [user0, user1], self.catalog, self.author, run_commit="a" * 40)
            self.assertEqual(result["actual_test_users"], 2)
            self.assertEqual(result["metrics"], dict.fromkeys(entry.METRICS, .5))
            self.assertEqual(result["anomalies"]["candidate_count"], 40)
            self.assertEqual(result["anomalies"]["complete_sid_legal_rate"], 1)
            self.assertEqual(result["prediction_wall_seconds_max_rank"], 1.0)
            with self.assertRaisesRegex(AssertionError, "identity"):
                entry.aggregate_predictions(paths, [user0, user1], self.catalog, self.author, run_commit="b" * 40)
            paths[1].write_text("")
            with self.assertRaisesRegex(AssertionError, "length"):
                entry.aggregate_predictions(paths, [user0, user1], self.catalog, self.author)
            paths[1].write_text(json.dumps({**row0, "rank": 1}) + "\n")
            with self.assertRaisesRegex(AssertionError, "Duplicate"):
                entry.aggregate_predictions(paths, [user0, user1], self.catalog, self.author)

    def test_bad_prediction_records_cannot_pass(self):
        row, user = self.prediction(0, "A", ["A"] + ["B"] * 19)
        for kind in ("target", "score", "parsed", "ordering", "beam_count"):
            broken = copy.deepcopy(row)
            if kind == "target":
                broken["target"] = "B"
            elif kind == "score":
                broken["candidates"][0]["sequence_score"] = float("nan")
            elif kind == "parsed":
                broken["candidates"][0]["sid"] = "B"
            elif kind == "ordering":
                broken["sorted_generation_order"] = list(reversed(range(20)))
            else:
                broken["candidates"].pop()
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                entry.audit_prediction(broken, user, self.catalog, self.author)


if __name__ == "__main__":
    unittest.main()
