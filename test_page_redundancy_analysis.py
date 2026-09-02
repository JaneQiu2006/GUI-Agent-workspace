from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from scripts.analyze_page_redundancy import (
    DHASH_THRESHOLDS,
    TILE_THRESHOLDS,
    analyze_records,
    available_groups,
    build_threshold_stats,
)
from test_framework.cache_inference import PageCacheConfig


class PageRedundancyAnalysisTest(unittest.TestCase):
    def test_streaming_history_grouping_and_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            first = image_dir / "first.png"
            duplicate = image_dir / "duplicate.png"
            cross_app = image_dir / "cross_app.png"
            self._write_image(first, marker=(255, 0, 0))
            duplicate.write_bytes(first.read_bytes())
            self._write_image(cross_app, marker=(0, 255, 0))

            samples = [
                {
                    "episode_id": "e1",
                    "step_id": 0,
                    "package": "pkg.same",
                    "goal": "task one",
                    "image_path": str(first.relative_to(root)).replace("\\", "/"),
                },
                {
                    "episode_id": "e1",
                    "step_id": 1,
                    "package": "pkg.same",
                    "goal": "task one",
                    "image_path": str(duplicate.relative_to(root)).replace("\\", "/"),
                },
                {
                    "episode_id": "e2",
                    "step_id": 0,
                    "package": "pkg.other",
                    "goal": "task two",
                    "image_path": str(cross_app.relative_to(root)).replace("\\", "/"),
                },
            ]
            test_json = root / "test.json"
            test_json.write_text(json.dumps({"samples": samples}), encoding="utf-8")

            selected = [(index, sample, 1 if sample["episode_id"] == "e1" else 2) for index, sample in enumerate(samples)]
            config = PageCacheConfig(mode="observe", scope="dataset", similarity="tile")
            details, metadata = analyze_records(selected, root, config, DHASH_THRESHOLDS, TILE_THRESHOLDS)
            groups = available_groups(metadata)
            threshold_stats = build_threshold_stats(groups, details, DHASH_THRESHOLDS, TILE_THRESHOLDS)

            self.assertFalse(details[0]["has_history"])
            self.assertTrue(details[1]["exact_sha256_match"])
            self.assertEqual(details[1]["nearest_global_index"], 1)
            self.assertTrue(details[1]["nearest_same_episode"])
            self.assertTrue(details[1]["nearest_same_app"])
            self.assertEqual(details[1]["current_cache_config_hit_type"], "exact")
            self.assertLessEqual(details[2]["nearest_global_index"], 2)
            self.assertFalse(details[2]["nearest_same_episode"])
            self.assertIn("same_episode", groups)
            self.assertIn("cross_app", groups)
            exact_overall = [
                row
                for row in threshold_stats
                if row["group"] == "overall" and row["threshold_type"] == "exact"
            ][0]
            self.assertEqual(exact_overall["hit_count"], 1)
            self.assertEqual(exact_overall["total_pages"], 3)

    def _write_image(self, path: Path, marker: tuple[int, int, int]) -> None:
        image = Image.new("RGB", (64, 64), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 24, 24), fill=marker)
        image.save(path)


if __name__ == "__main__":
    unittest.main()
