from __future__ import annotations

import unittest

from scripts.analyze_visual_feature_locality import (
    align_changed_tiles_to_tokens,
    build_distance_profile,
    feature_distances,
    grouped_by,
    infer_token_grid,
)


class VisualFeatureLocalityAnalysisTest(unittest.TestCase):
    def test_align_changed_tiles_to_different_visual_token_grid(self) -> None:
        changed_mask = [
            True, False, False, False,
            False, False, False, False,
        ]
        token_mask = align_changed_tiles_to_tokens(
            changed_mask,
            total_tile_count=8,
            changed_tile_count=1,
            tile_rows=2,
            tile_cols=4,
            token_count=16,
            token_grid=(4, 4),
        )
        self.assertEqual(len(token_mask), 16)
        self.assertEqual(sum(token_mask), 2)
        self.assertTrue(all(token_mask[index] for index in (0, 4)))

    def test_distance_profile_bins_by_manhattan_distance(self) -> None:
        metrics = {
            "cosine_distance": [1.0, 0.5, 0.25, 0.125],
            "relative_l2_distance": [2.0, 1.0, 0.5, 0.25],
        }
        profile = build_distance_profile(metrics, [True, False, False, False], (2, 2))
        by_distance = {row["token_manhattan_distance"]: row for row in profile}
        self.assertEqual(by_distance[0]["token_count"], 1)
        self.assertEqual(by_distance[1]["token_count"], 2)
        self.assertEqual(by_distance[2]["token_count"], 1)
        self.assertAlmostEqual(by_distance[1]["cosine_distance"]["mean"], 0.375)

    def test_feature_distances_and_grid_inference(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in the local static-check environment")

        prev = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        cur = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        cosine, relative_l2 = feature_distances(prev, cur)
        self.assertAlmostEqual(float(cosine[0]), 0.0)
        self.assertAlmostEqual(float(cosine[1]), 0.0)
        self.assertAlmostEqual(float(relative_l2[0]), 0.0)
        self.assertAlmostEqual(float(relative_l2[1]), 0.5)

        grid = infer_token_grid({"image_grid_thw": [[1, 8, 8]], "processor_merge_size": 2}, 16)
        self.assertEqual(grid, (4, 4))

    def test_grouped_stats_keep_thresholds_separate(self) -> None:
        rows = [
            {"app": "calendar", "layer_id": "final", "threshold": 0.01, "feature_metric": "cosine_distance", "pixel_changed_ratio": 0.1, "feature_changed_ratio": 0.2},
            {"app": "calendar", "layer_id": "final", "threshold": 0.10, "feature_metric": "cosine_distance", "pixel_changed_ratio": 0.1, "feature_changed_ratio": 0.05},
        ]
        grouped = grouped_by(rows, "app")
        self.assertEqual(len(grouped), 2)
        self.assertEqual({row["threshold"] for row in grouped}, {"0.01", "0.1"})


if __name__ == "__main__":
    unittest.main()
