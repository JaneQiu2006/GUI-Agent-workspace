from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
TEST_FRAMEWORK = REPO_ROOT / "test_framework"
for path in (SCRIPT_DIR, TEST_FRAMEWORK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_kv_locality import build_parser, kv_tensor_to_token_matrix, required_output_fields  # noqa: E402
from analyze_visual_feature_locality import align_changed_tiles_to_tokens  # noqa: E402
from hf_gui_baseline import infer_visual_token_positions  # noqa: E402


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class FakeTokenizer:
    image_token_id = 999


class FakeProcessor:
    tokenizer = FakeTokenizer()


class KVLocalityHelpersTest(unittest.TestCase):
    def test_infer_visual_token_positions_from_image_token_id(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed")
        info = infer_visual_token_positions(
            FakeProcessor(),
            {
                "input_ids": torch.tensor([[11, 22, 999, 999, 999, 33]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            },
        )
        self.assertEqual(info.image_token_id, 999)
        self.assertEqual(info.visual_token_positions, [2, 3, 4])
        self.assertEqual(info.visual_token_span, (2, 5))
        self.assertEqual(info.text_token_positions, [0, 1, 5])
        self.assertEqual(info.visual_position_source, "input_ids_image_token_id")

    def test_kv_tensor_to_token_matrix_normalizes_common_shapes(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed")
        heads_seq = torch.randn(1, 2, 5, 3)
        seq_heads = torch.randn(1, 5, 2, 3)
        no_batch = torch.randn(2, 5, 3)
        self.assertEqual(tuple(kv_tensor_to_token_matrix(heads_seq, expected_seq_len=5).shape), (5, 6))
        self.assertEqual(tuple(kv_tensor_to_token_matrix(seq_heads, expected_seq_len=5).shape), (5, 6))
        self.assertEqual(tuple(kv_tensor_to_token_matrix(no_batch, expected_seq_len=5).shape), (5, 6))

    def test_tile_mask_alignment_reuses_visual_locality_logic(self) -> None:
        mask = align_changed_tiles_to_tokens(
            changed_tile_mask=[True, False, False, False],
            total_tile_count=4,
            changed_tile_count=1,
            tile_rows=2,
            tile_cols=2,
            token_count=16,
            token_grid=(4, 4),
        )
        self.assertEqual(len(mask), 16)
        self.assertEqual(sum(mask), 4)
        self.assertEqual(mask[:2], [True, True])
        self.assertEqual(mask[4:6], [True, True])

    def test_cli_parse_and_required_output_fields(self) -> None:
        args = build_parser().parse_args(
            [
                "--prev_image",
                "a.png",
                "--cur_image",
                "b.png",
                "--kv_scope",
                "visual_tokens",
                "--kv_kind",
                "key",
                "--kv_threshold",
                "0.01",
            ]
        )
        self.assertEqual(str(args.prev_image), "a.png")
        self.assertEqual(args.kv_scopes, ["visual_tokens"])
        self.assertEqual(args.kv_kinds, ["key"])
        self.assertEqual(args.kv_thresholds, [0.01])
        fields = set(required_output_fields())
        for field in (
            "kv_boundary",
            "kv_scope",
            "layer_id",
            "kv_kind",
            "kv_changed_ratio",
            "visual_position_source",
            "distance_profile",
        ):
            self.assertIn(field, fields)


if __name__ == "__main__":
    unittest.main()
