"""Page-level cache helpers for the HuggingFace GUI baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Optional, Tuple

from cache_fingerprint import (
    PageFingerprint,
    PageSimilarity,
    compare_page_fingerprints,
    compute_page_fingerprint,
)
from cache_store import LruCache


PAGE_CACHE_MODES = ("off", "observe", "inputs")
PAGE_CACHE_SCOPES = ("trajectory", "dataset", "session")
PAGE_CACHE_SIMILARITIES = ("exact", "dhash", "tile")


@dataclass
class PageCacheConfig:
    mode: str = "off"
    scope: str = "trajectory"
    similarity: str = "tile"
    max_entries: int = 128
    near_dhash_threshold: int = 4
    patch_tile_threshold: float = 0.90
    patch_max_changed_area_ratio: float = 0.25
    patch_critical_regions: Tuple[Tuple[float, float, float, float], ...] = ()
    near_tile_threshold: float = 0.98
    tile_rows: int = 8
    tile_cols: int = 16
    ignored_top_ratio: float = 0.0
    ignored_bottom_ratio: float = 0.0
    identity: str = ""

    def __post_init__(self) -> None:
        if self.mode not in PAGE_CACHE_MODES:
            raise ValueError(f"Unsupported page cache mode: {self.mode}")
        if self.scope not in PAGE_CACHE_SCOPES:
            raise ValueError(f"Unsupported page cache scope: {self.scope}")
        if self.similarity not in PAGE_CACHE_SIMILARITIES:
            raise ValueError(f"Unsupported page cache similarity: {self.similarity}")
        if not 0.0 <= self.patch_max_changed_area_ratio <= 1.0:
            raise ValueError("page cache patch_max_changed_area_ratio must be in [0, 1]")
        self.patch_critical_regions = tuple(
            _normalize_bbox(region) for region in self.patch_critical_regions
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageCacheProbe:
    mode: str
    page_cache_hit_type: str
    page_cache_hit: bool
    processor_cache_hit: bool
    similarity_dhash_hamming: Optional[int]
    tile_unchanged_ratio: Optional[float]
    changed_tile_count: int
    changed_bbox: Optional[Any]
    changed_bbox_pixels: Optional[Any]
    changed_bbox_area_ratio: float
    cache_lookup_seconds: float
    patch_candidate: bool = False
    patch_candidate_allowed: bool = False
    patch_risk_reasons: Tuple[str, ...] = ()
    patch_diff: Optional[Dict[str, Any]] = None
    cache_write_seconds: float = 0.0
    cache_entries: int = 0
    cache_evictions: int = 0
    processor_cache_entries: int = 0
    page_cache_entries: int = 0
    image_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.changed_bbox is not None and not isinstance(self.changed_bbox, list):
            data["changed_bbox"] = list(self.changed_bbox)
        if self.changed_bbox_pixels is not None and not isinstance(self.changed_bbox_pixels, list):
            data["changed_bbox_pixels"] = list(self.changed_bbox_pixels)
        return data


@dataclass
class _PageRecord:
    fingerprint: PageFingerprint
    trajectory_id: Optional[str]


class PageLevelCache:
    """Exact processor-input cache plus page similarity observer."""

    def __init__(self, config: PageCacheConfig) -> None:
        self.config = config
        self._pages: LruCache[str, _PageRecord] = LruCache(config.max_entries)
        self._processor_inputs: LruCache[str, Any] = LruCache(config.max_entries)
        self._active_trajectory_id: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    def begin_step(self, image_path: Path, trajectory_id: Optional[Any] = None) -> tuple[PageFingerprint, PageCacheProbe]:
        started = time.perf_counter()
        trajectory_text = None if trajectory_id is None else str(trajectory_id)
        self._maybe_reset_scope(trajectory_text)
        fingerprint = compute_page_fingerprint(
            image_path,
            tile_rows=self.config.tile_rows,
            tile_cols=self.config.tile_cols,
            ignored_top_ratio=self.config.ignored_top_ratio,
            ignored_bottom_ratio=self.config.ignored_bottom_ratio,
        )
        hit_type, similarity, base_record = self._best_page_match(fingerprint, trajectory_text)
        lookup_seconds = time.perf_counter() - started
        probe = self._probe_from_match(
            fingerprint=fingerprint,
            hit_type=hit_type,
            similarity=similarity,
            base_record=base_record,
            lookup_seconds=lookup_seconds,
            processor_cache_hit=False,
        )
        return fingerprint, probe

    def finish_step(self, fingerprint: PageFingerprint, trajectory_id: Optional[Any], probe: PageCacheProbe) -> None:
        started = time.perf_counter()
        trajectory_text = None if trajectory_id is None else str(trajectory_id)
        self._pages.put(fingerprint.image_sha256, _PageRecord(fingerprint, trajectory_text))
        probe.cache_write_seconds += time.perf_counter() - started
        self._fill_counts(probe)

    def get_processor_inputs(self, key: str) -> Optional[Any]:
        if self.config.mode != "inputs":
            return None
        return self._clone_inputs(self._processor_inputs.get(key))

    def put_processor_inputs(self, key: str, inputs: Any) -> None:
        if self.config.mode != "inputs":
            return
        self._processor_inputs.put(key, self._clone_inputs(inputs))

    def processor_key(
        self,
        chat_text: str,
        fingerprint: PageFingerprint,
        visual_token_mode: str,
        min_pixels: Optional[int],
        max_pixels: Optional[int],
    ) -> str:
        payload = "|".join(
            [
                self.config.identity,
                visual_token_mode,
                str(min_pixels),
                str(max_pixels),
                str(fingerprint.width),
                str(fingerprint.height),
                fingerprint.image_sha256,
                hashlib.sha256(chat_text.encode("utf-8")).hexdigest(),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.config.mode,
            "scope": self.config.scope,
            "similarity": self.config.similarity,
            "page_cache_entries": len(self._pages),
            "processor_cache_entries": len(self._processor_inputs),
            "cache_evictions": self._pages.evictions + self._processor_inputs.evictions,
        }

    def clear(self) -> None:
        self._pages.clear()
        self._processor_inputs.clear()
        self._active_trajectory_id = None

    def _maybe_reset_scope(self, trajectory_id: Optional[str]) -> None:
        if self.config.scope != "trajectory":
            return
        if self._active_trajectory_id is None:
            self._active_trajectory_id = trajectory_id
            return
        if trajectory_id != self._active_trajectory_id:
            self._pages.clear()
            self._processor_inputs.clear()
            self._active_trajectory_id = trajectory_id

    def _best_page_match(
        self,
        fingerprint: PageFingerprint,
        trajectory_id: Optional[str],
    ) -> tuple[str, Optional[PageSimilarity], Optional[_PageRecord]]:
        exact_record = self._pages.get(fingerprint.image_sha256)
        if exact_record is not None and self._record_in_scope(exact_record, trajectory_id):
            return "exact", compare_page_fingerprints(fingerprint, exact_record.fingerprint), exact_record

        best_similarity: Optional[PageSimilarity] = None
        best_record: Optional[_PageRecord] = None
        best_score = (-1.0, -10_000.0)
        for _, record in self._pages.items():
            if not self._record_in_scope(record, trajectory_id):
                continue
            similarity = compare_page_fingerprints(fingerprint, record.fingerprint)
            tile_ratio = similarity.tile_unchanged_ratio or 0.0
            dhash_score = -float(similarity.dhash_hamming or 10_000)
            score = (tile_ratio, dhash_score)
            if score > best_score:
                best_score = score
                best_similarity = similarity
                best_record = record
        if best_similarity is None:
            return "miss", None, None
        if self._near_hit(best_similarity):
            return "near", best_similarity, best_record
        if self._patch_candidate(best_similarity):
            return "patch_candidate", best_similarity, best_record
        return "miss", best_similarity, best_record

    def _record_in_scope(self, record: _PageRecord, trajectory_id: Optional[str]) -> bool:
        if self.config.scope != "trajectory":
            return True
        return record.trajectory_id == trajectory_id

    def _near_hit(self, similarity: PageSimilarity) -> bool:
        if self.config.similarity == "exact":
            return False
        dhash_ok = (
            similarity.dhash_hamming is not None
            and similarity.dhash_hamming <= self.config.near_dhash_threshold
        )
        tile_ok = (
            similarity.tile_unchanged_ratio is not None
            and similarity.tile_unchanged_ratio >= self.config.near_tile_threshold
        )
        if self.config.similarity == "dhash":
            return dhash_ok
        return dhash_ok and tile_ok

    def _patch_candidate(self, similarity: PageSimilarity) -> bool:
        if self.config.similarity not in {"tile", "dhash"}:
            return False
        if similarity.exact or similarity.changed_tile_count <= 0:
            return False
        if self._patch_risk_reasons(similarity):
            return False
        return (
            similarity.tile_unchanged_ratio is not None
            and similarity.tile_unchanged_ratio >= self.config.patch_tile_threshold
        )

    def _raw_patch_candidate(self, similarity: PageSimilarity) -> bool:
        return (
            self.config.similarity in {"tile", "dhash"}
            and not similarity.exact
            and similarity.changed_tile_count > 0
            and similarity.tile_unchanged_ratio is not None
            and similarity.tile_unchanged_ratio >= self.config.patch_tile_threshold
        )

    def _patch_risk_reasons(self, similarity: PageSimilarity) -> Tuple[str, ...]:
        reasons = []
        if similarity.changed_bbox_area_ratio > self.config.patch_max_changed_area_ratio:
            reasons.append("changed_bbox_area_too_large")
        changed_bbox = similarity.changed_bbox
        if changed_bbox is not None:
            for region in self.config.patch_critical_regions:
                if _bbox_intersects(changed_bbox, region):
                    reasons.append("critical_region_overlap")
                    break
        return tuple(reasons)

    def _probe_from_match(
        self,
        fingerprint: PageFingerprint,
        hit_type: str,
        similarity: Optional[PageSimilarity],
        base_record: Optional[_PageRecord],
        lookup_seconds: float,
        processor_cache_hit: bool,
    ) -> PageCacheProbe:
        patch_candidate = self._raw_patch_candidate(similarity) if similarity else False
        patch_risk_reasons = self._patch_risk_reasons(similarity) if similarity else ()
        patch_candidate_allowed = patch_candidate and not patch_risk_reasons
        probe = PageCacheProbe(
            mode=self.config.mode,
            page_cache_hit_type=hit_type,
            page_cache_hit=hit_type in {"exact", "near", "patch_candidate"},
            processor_cache_hit=processor_cache_hit,
            similarity_dhash_hamming=similarity.dhash_hamming if similarity else None,
            tile_unchanged_ratio=similarity.tile_unchanged_ratio if similarity else None,
            changed_tile_count=similarity.changed_tile_count if similarity else 0,
            changed_bbox=similarity.changed_bbox if similarity else None,
            changed_bbox_pixels=similarity.changed_bbox_pixels if similarity else None,
            changed_bbox_area_ratio=similarity.changed_bbox_area_ratio if similarity else 0.0,
            patch_candidate=patch_candidate,
            patch_candidate_allowed=patch_candidate_allowed,
            patch_risk_reasons=patch_risk_reasons,
            patch_diff=self._patch_diff_dict(fingerprint, similarity, base_record),
            cache_lookup_seconds=lookup_seconds,
            image_sha256=fingerprint.image_sha256,
        )
        self._fill_counts(probe)
        return probe

    def _patch_diff_dict(
        self,
        fingerprint: PageFingerprint,
        similarity: Optional[PageSimilarity],
        base_record: Optional[_PageRecord],
    ) -> Optional[Dict[str, Any]]:
        if (
            similarity is None
            or similarity.exact
            or similarity.changed_tile_count <= 0
            or similarity.tile_unchanged_ratio is None
        ):
            return None
        return {
            "base_image_sha256": base_record.fingerprint.image_sha256 if base_record else None,
            "current_image_sha256": fingerprint.image_sha256,
            "tile_rows": fingerprint.tile_rows,
            "tile_cols": fingerprint.tile_cols,
            "total_tile_count": similarity.total_tile_count,
            "unchanged_tile_count": similarity.unchanged_tile_count,
            "changed_tile_count": similarity.changed_tile_count,
            "tile_unchanged_ratio": similarity.tile_unchanged_ratio,
            "changed_tile_indices": list(similarity.changed_tile_indices),
            "unchanged_tile_indices": list(similarity.unchanged_tile_indices),
            "changed_tile_mask": list(similarity.changed_tile_mask),
            "stable_tile_hashes": [
                {"index": index, "hash": value}
                for index, value in similarity.stable_tile_hashes
            ],
            "changed_bbox": list(similarity.changed_bbox) if similarity.changed_bbox else None,
            "changed_bbox_pixels": list(similarity.changed_bbox_pixels) if similarity.changed_bbox_pixels else None,
            "changed_bbox_area_ratio": similarity.changed_bbox_area_ratio,
            "dhash_hamming": similarity.dhash_hamming,
        }

    def _fill_counts(self, probe: PageCacheProbe) -> None:
        probe.page_cache_entries = len(self._pages)
        probe.processor_cache_entries = len(self._processor_inputs)
        probe.cache_entries = probe.page_cache_entries + probe.processor_cache_entries
        probe.cache_evictions = self._pages.evictions + self._processor_inputs.evictions

    def _clone_inputs(self, inputs: Any) -> Any:
        if inputs is None:
            return None
        if isinstance(inputs, dict):
            data = dict(inputs)
        elif hasattr(inputs, "items"):
            data = dict(inputs.items())
        else:
            return inputs
        for key, value in list(data.items()):
            if hasattr(value, "detach") and hasattr(value, "clone"):
                data[key] = value.detach().clone().cpu()
            elif hasattr(value, "clone"):
                data[key] = value.clone()
        if isinstance(inputs, dict):
            return data
        try:
            return inputs.__class__(data=data)
        except TypeError:
            try:
                return inputs.__class__(data)
            except TypeError:
                return data


def parse_normalized_bboxes(values: Iterable[str]) -> Tuple[Tuple[float, float, float, float], ...]:
    return tuple(_normalize_bbox(_parse_bbox_text(value)) for value in values)


def _parse_bbox_text(value: str) -> Tuple[float, float, float, float]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected bbox as left,top,right,bottom, got: {value}")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def _normalize_bbox(region: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    left, top, right, bottom = region
    left = max(0.0, min(1.0, float(left)))
    top = max(0.0, min(1.0, float(top)))
    right = max(0.0, min(1.0, float(right)))
    bottom = max(0.0, min(1.0, float(bottom)))
    return (
        min(left, right),
        min(top, bottom),
        max(left, right),
        max(top, bottom),
    )


def _bbox_intersects(
    left_bbox: Tuple[float, float, float, float],
    right_bbox: Tuple[float, float, float, float],
) -> bool:
    left_a, top_a, right_a, bottom_a = left_bbox
    left_b, top_b, right_b, bottom_b = right_bbox
    return max(left_a, left_b) < min(right_a, right_b) and max(top_a, top_b) < min(bottom_a, bottom_b)
