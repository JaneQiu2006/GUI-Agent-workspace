"""Page-level screenshot fingerprints for GUI inference cache experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PageFingerprint:
    image_sha256: str
    width: int
    height: int
    dhash64: str
    tile_rows: int
    tile_cols: int
    tile_hashes: Tuple[str, ...]
    ignored_top_ratio: float = 0.0
    ignored_bottom_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tile_hashes"] = list(self.tile_hashes)
        return data


@dataclass(frozen=True)
class PageSimilarity:
    exact: bool
    dhash_hamming: Optional[int]
    tile_unchanged_ratio: Optional[float]
    total_tile_count: int
    unchanged_tile_count: int
    changed_tile_count: int
    changed_tile_indices: Tuple[int, ...]
    unchanged_tile_indices: Tuple[int, ...]
    changed_tile_mask: Tuple[bool, ...]
    stable_tile_hashes: Tuple[Tuple[int, str], ...]
    changed_bbox: Optional[Tuple[float, float, float, float]]
    changed_bbox_pixels: Optional[Tuple[int, int, int, int]]
    changed_bbox_area_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.changed_bbox is not None:
            data["changed_bbox"] = list(self.changed_bbox)
        if self.changed_bbox_pixels is not None:
            data["changed_bbox_pixels"] = list(self.changed_bbox_pixels)
        return data


def compute_page_fingerprint(
    image_path: Path,
    tile_rows: int = 8,
    tile_cols: int = 16,
    ignored_top_ratio: float = 0.0,
    ignored_bottom_ratio: float = 0.0,
) -> PageFingerprint:
    """Compute exact, perceptual, and tile-level fingerprints for one screenshot."""
    from PIL import Image

    image_bytes = Path(image_path).read_bytes()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    active = _active_image(image, ignored_top_ratio, ignored_bottom_ratio)
    return PageFingerprint(
        image_sha256=image_sha256,
        width=width,
        height=height,
        dhash64=_dhash64(active),
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tile_hashes=tuple(_tile_hashes(active, tile_rows, tile_cols)),
        ignored_top_ratio=float(ignored_top_ratio),
        ignored_bottom_ratio=float(ignored_bottom_ratio),
    )


def compare_page_fingerprints(
    current: PageFingerprint,
    cached: PageFingerprint,
) -> PageSimilarity:
    exact = current.image_sha256 == cached.image_sha256
    dhash_hamming = hamming_hex(current.dhash64, cached.dhash64)
    tile_ratio: Optional[float] = None
    changed_count = 0
    changed_bbox: Optional[Tuple[float, float, float, float]] = None
    area_ratio = 0.0
    if _compatible_tiles(current, cached):
        changed = tuple(
            index
            for index, (left, right) in enumerate(zip(current.tile_hashes, cached.tile_hashes))
            if left != right
        )
        unchanged = tuple(
            index
            for index, (left, right) in enumerate(zip(current.tile_hashes, cached.tile_hashes))
            if left == right
        )
        changed_count = len(changed)
        total = max(1, len(current.tile_hashes))
        tile_ratio = (total - changed_count) / total
        active_bbox = _changed_tile_bbox(changed, current.tile_rows, current.tile_cols)
        changed_bbox = _map_active_bbox_to_image(
            active_bbox,
            current.ignored_top_ratio,
            current.ignored_bottom_ratio,
        )
        changed_bbox_pixels = _bbox_pixels(changed_bbox, current.width, current.height)
        area_ratio = _bbox_area(changed_bbox)
        changed_tile_indices = changed
        unchanged_tile_indices = unchanged
        changed_set = set(changed)
        changed_tile_mask = tuple(index in changed_set for index in range(len(current.tile_hashes)))
        stable_tile_hashes = tuple((index, current.tile_hashes[index]) for index in unchanged)
    else:
        changed_tile_indices = ()
        unchanged_tile_indices = ()
        changed_tile_mask = ()
        stable_tile_hashes = ()
        changed_bbox_pixels = None
    return PageSimilarity(
        exact=exact,
        dhash_hamming=dhash_hamming,
        tile_unchanged_ratio=tile_ratio,
        total_tile_count=len(current.tile_hashes) if _compatible_tiles(current, cached) else 0,
        unchanged_tile_count=len(unchanged_tile_indices),
        changed_tile_count=changed_count,
        changed_tile_indices=changed_tile_indices,
        unchanged_tile_indices=unchanged_tile_indices,
        changed_tile_mask=changed_tile_mask,
        stable_tile_hashes=stable_tile_hashes,
        changed_bbox=changed_bbox,
        changed_bbox_pixels=changed_bbox_pixels,
        changed_bbox_area_ratio=area_ratio,
    )


def hamming_hex(left: str, right: str) -> Optional[int]:
    if not left or not right or len(left) != len(right):
        return None
    return (int(left, 16) ^ int(right, 16)).bit_count()


def image_file_sha256(image_path: Path) -> str:
    return hashlib.sha256(Path(image_path).read_bytes()).hexdigest()


def _active_image(image: Any, ignored_top_ratio: float, ignored_bottom_ratio: float) -> Any:
    width, height = image.size
    top = int(max(0.0, min(0.5, ignored_top_ratio)) * height)
    bottom_crop = int(max(0.0, min(0.5, ignored_bottom_ratio)) * height)
    bottom = max(top + 1, height - bottom_crop)
    return image.crop((0, top, width, bottom))


def _dhash64(image: Any) -> str:
    small = image.convert("L").resize((9, 8))
    pixels = list(small.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value <<= 1
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            if left > right:
                value |= 1
    return f"{value:016x}"


def _tile_hashes(image: Any, rows: int, cols: int) -> List[str]:
    width, height = image.size
    hashes = []
    for row in range(rows):
        for col in range(cols):
            left = round(col * width / cols)
            upper = round(row * height / rows)
            right = round((col + 1) * width / cols)
            lower = round((row + 1) * height / rows)
            tile = image.crop((left, upper, max(left + 1, right), max(upper + 1, lower)))
            normalized = tile.convert("L").resize((8, 8))
            hashes.append(hashlib.sha1(bytes(normalized.getdata())).hexdigest()[:16])
    return hashes


def _compatible_tiles(left: PageFingerprint, right: PageFingerprint) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and left.tile_rows == right.tile_rows
        and left.tile_cols == right.tile_cols
        and len(left.tile_hashes) == len(right.tile_hashes)
        and left.ignored_top_ratio == right.ignored_top_ratio
        and left.ignored_bottom_ratio == right.ignored_bottom_ratio
    )


def _changed_tile_bbox(
    changed_indices: Sequence[int],
    rows: int,
    cols: int,
) -> Optional[Tuple[float, float, float, float]]:
    if not changed_indices:
        return None
    changed_rows = [index // cols for index in changed_indices]
    changed_cols = [index % cols for index in changed_indices]
    left = min(changed_cols) / cols
    top = min(changed_rows) / rows
    right = (max(changed_cols) + 1) / cols
    bottom = (max(changed_rows) + 1) / rows
    return (left, top, right, bottom)


def _bbox_area(bbox: Optional[Tuple[float, float, float, float]]) -> float:
    if bbox is None:
        return 0.0
    left, top, right, bottom = bbox
    return max(0.0, right - left) * max(0.0, bottom - top)


def _map_active_bbox_to_image(
    bbox: Optional[Tuple[float, float, float, float]],
    ignored_top_ratio: float,
    ignored_bottom_ratio: float,
) -> Optional[Tuple[float, float, float, float]]:
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    active_top = max(0.0, min(0.5, ignored_top_ratio))
    active_bottom = 1.0 - max(0.0, min(0.5, ignored_bottom_ratio))
    active_height = max(0.0, active_bottom - active_top)
    return (
        left,
        active_top + top * active_height,
        right,
        active_top + bottom * active_height,
    )


def _bbox_pixels(
    bbox: Optional[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return (
        int(round(left * width)),
        int(round(top * height)),
        int(round(right * width)),
        int(round(bottom * height)),
    )
