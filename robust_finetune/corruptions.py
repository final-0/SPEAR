"""Corruption synthesis for SPEAR stage 2, matched to the RobustSpring benchmark.

frost, rain and snow add layers extracted from RobustSpring (``WEATHER_OVERLAY_DIR``); frost uses
one static layer per triplet, rain/snow an independent layer per frame.  Noise corruptions use an
independent seed per frame, the other appearance corruptions share one seed.  elastic_transform
warps the triplet and transforms the flow labels consistently.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from imagecorruptions import corruptions as image_c


# imagecorruptions 1.1.2 still passes the removed ``multichannel`` keyword.
# Translate it for current scikit-image without downgrading the environment.
_skimage_gaussian = image_c.gaussian


def _gaussian_compat(*args, **kwargs):
    multichannel = kwargs.pop("multichannel", None)
    if multichannel is not None:
        kwargs["channel_axis"] = -1 if multichannel else None
    return _skimage_gaussian(*args, **kwargs)


image_c.gaussian = _gaussian_compat


CORRUPTION_GROUPS = {
    "color": ("brightness", "contrast", "saturate"),
    "blur": (
        "defocus_blur",
        "gaussian_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
    ),
    "noise": (
        "gaussian_noise",
        "impulse_noise",
        "speckle_noise",
        "shot_noise",
    ),
    "quality": ("pixelate", "jpeg_compression", "elastic_transform"),
    "weather": ("fog", "spatter", "frost", "snow", "rain"),
}

ROBUSTSPRING_CORRUPTIONS = tuple(
    name for names in CORRUPTION_GROUPS.values() for name in names
)
NOISE_CORRUPTIONS = frozenset(CORRUPTION_GROUPS["noise"])
# Expert routing: the router emits one INDEPENDENT sigmoid gate per group (not a softmax).
# Clean frames and corruptions not listed here ("other") get an all-zero target, so every gate stays closed.
ROUTER_EXPERT_GROUPS = ("frost", "weather_other", "quality")
CORRUPTION_TO_EXPERT_GROUP = {
    "frost": "frost",
    "fog": "weather_other",
    "spatter": "weather_other",
    "snow": "weather_other",
    "rain": "weather_other",
    "jpeg_compression": "quality",
    "elastic_transform": "quality",
    "pixelate": "quality",
}
EXPERT_GROUP_INDEX = {
    name: index for index, name in enumerate(ROUTER_EXPERT_GROUPS)
}
NUM_EXPERT_GROUPS = len(ROUTER_EXPERT_GROUPS)


def expert_group_label(name):
    """Router target over ROUTER_EXPERT_GROUPS; all-zero for clean (None) and "other" corruptions."""
    label = [0.0] * NUM_EXPERT_GROUPS
    if name is None:
        return label
    group = CORRUPTION_TO_EXPERT_GROUP.get(name)
    if group is not None:
        label[EXPERT_GROUP_INDEX[group]] = 1.0
    return label

# Training severity ranges (inclusive), calibrated so the mean synthetic magnitude matches the
# RobustSpring test images of each corruption.
DEFAULT_SEVERITY_RANGE = (1, 5)
SEVERITY_RANGES = {
    "jpeg_compression": (4, 5),
    "fog": (4, 5),
    "pixelate": (4, 5),
    "brightness": (3, 4),
    "speckle_noise": (3, 4),
    "gaussian_noise": (1, 2),
    "shot_noise": (1, 2),
    "elastic_transform": (1, 2),
    "impulse_noise": (2, 3),
}


def sample_severity(name, rng):
    low, high = SEVERITY_RANGES.get(name, DEFAULT_SEVERITY_RANGE)
    return rng.randint(low, high)


def _frame_seed(base_seed: int, frame_index: int) -> int:
    """Mix an absolute frame index into a reproducible unsigned 32-bit seed."""
    value = int(frame_index) & 0xFFFFFFFF
    value = (value + 0x9E3779B9) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x85EBCA6B) & 0xFFFFFFFF
    value ^= value >> 13
    return (int(base_seed) ^ value) & 0xFFFFFFFF


def _glass_blur(image: Image.Image, severity: int = 1) -> np.ndarray:
    """Fast, replayable approximation of glass blur for online training.

    The reference imagecorruptions implementation performs Python-level pixel
    swaps and takes tens of seconds per 540x960 pair.  A smooth random remap
    retains the local refraction-plus-blur effect while keeping data loading
    fast enough for synchronous multi-GPU training.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = arr.shape[:2]
    blur_sigma = (0.7, 0.9, 1.1, 1.3, 1.5)[severity - 1]
    displacement = (0.7, 1.0, 1.3, 1.7, 2.1)[severity - 1]
    field_sigma = (1.5, 1.8, 2.1, 2.5, 3.0)[severity - 1]

    dx = np.random.standard_normal((height, width)).astype(np.float32)
    dy = np.random.standard_normal((height, width)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), field_sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), field_sigma)
    dx *= displacement / max(float(dx.std()), 1e-6)
    dy *= displacement / max(float(dy.std()), 1e-6)

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    warped = cv2.remap(
        arr,
        grid_x + dx,
        grid_y + dy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return cv2.GaussianBlur(warped, (0, 0), blur_sigma)


def _impulse_noise(image: Image.Image, severity: int = 1) -> np.ndarray:
    """Deterministic salt-and-pepper noise at imagecorruptions strengths."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    amount = (0.03, 0.06, 0.09, 0.17, 0.27)[severity - 1]
    draw = np.random.random(arr.shape[:2])
    arr[draw < amount / 2.0] = 0
    arr[(draw >= amount / 2.0) & (draw < amount)] = 255
    return arr


def _elastic_field(
    height: int,
    width: int,
    severity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create two smoothly evolving inverse-remap displacement fields."""
    amplitude = (1.25, 2.0, 3.0, 4.25, 5.75)[severity - 1]
    field_sigma = (18.0, 16.0, 14.0, 12.0, 10.0)[severity - 1]
    temporal_change = (0.03, 0.05, 0.07, 0.10, 0.13)[severity - 1]

    fields = []
    for _ in range(4):
        field = np.random.standard_normal((height, width)).astype(np.float32)
        field = cv2.GaussianBlur(field, (0, 0), field_sigma)
        field *= amplitude / max(float(field.std()), 1e-6)
        fields.append(field)

    dx0, dy0, dx_delta, dy_delta = fields
    dx1 = (1.0 - temporal_change) * dx0 + temporal_change * dx_delta
    dy1 = (1.0 - temporal_change) * dy0 + temporal_change * dy_delta
    return dx0, dy0, dx1, dy1


def _invert_inverse_map(
    point_x: np.ndarray,
    point_y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve q + d(q) = p for q by fixed-point iteration."""
    query_x = point_x.copy()
    query_y = point_y.copy()
    for _ in range(iterations):
        sampled_dx = cv2.remap(
            dx,
            query_x,
            query_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        sampled_dy = cv2.remap(
            dy,
            query_x,
            query_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        query_x = point_x - sampled_dx
        query_y = point_y - sampled_dy
    return query_x, query_y


def _warp_image_with_inverse_field(
    image: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return np.ascontiguousarray(
        cv2.remap(
            np.asarray(image, dtype=np.uint8),
            grid_x + dx,
            grid_y + dy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    )


def _transform_flow_between_inverse_fields(
    flow: np.ndarray,
    valid: np.ndarray,
    source_dx: np.ndarray,
    source_dy: np.ndarray,
    target_dx: np.ndarray,
    target_dy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform flow according to independently warped source/target frames."""
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    source_map_x = grid_x + source_dx
    source_map_y = grid_y + source_dy
    clean_flow = np.asarray(flow, dtype=np.float32)
    finite = np.isfinite(clean_flow).all(axis=2)
    clean_flow = np.where(np.isfinite(clean_flow), clean_flow, 0.0)
    sampled_flow = cv2.remap(
        clean_flow,
        source_map_x,
        source_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled_valid = cv2.remap(
        (np.asarray(valid) >= 0.5).astype(np.uint8),
        source_map_x,
        source_map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    sampled_finite = cv2.remap(
        finite.astype(np.uint8),
        source_map_x,
        source_map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

    target_clean_x = source_map_x + sampled_flow[..., 0]
    target_clean_y = source_map_y + sampled_flow[..., 1]
    target_x, target_y = _invert_inverse_map(
        target_clean_x,
        target_clean_y,
        target_dx,
        target_dy,
    )
    output_flow = np.stack((target_x - grid_x, target_y - grid_y), axis=-1)
    source_inside = (
        (source_map_x >= 0)
        & (source_map_x <= width - 1)
        & (source_map_y >= 0)
        & (source_map_y <= height - 1)
    )
    target_inside = (
        (target_clean_x >= 0)
        & (target_clean_x <= width - 1)
        & (target_clean_y >= 0)
        & (target_clean_y <= height - 1)
        & (target_x >= 0)
        & (target_x <= width - 1)
        & (target_y >= 0)
        & (target_y <= height - 1)
    )
    output_valid = sampled_valid & sampled_finite & source_inside & target_inside
    return (
        np.ascontiguousarray(output_flow.astype(np.float32)),
        np.ascontiguousarray(output_valid.astype(np.float32)),
    )


def _elastic_triplet(
    images: list[np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray],
    backward_flow: np.ndarray,
    backward_valid: np.ndarray,
    forward_flow: np.ndarray,
    forward_valid: np.ndarray,
    severity: int,
    seed: int,
) -> tuple[
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Warp t-1/t/t+1 and both center-originating labels consistently."""
    if len(images) != 3:
        raise ValueError("elastic triplet requires exactly three images")
    height, width = images[1].shape[:2]
    if any(image.shape[:2] != (height, width) for image in images):
        raise ValueError("elastic triplet images must have identical shapes")
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        center_dx, center_dy, next_dx, next_dy = _elastic_field(
            height, width, severity
        )
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
    previous_dx = center_dx - (next_dx - center_dx)
    previous_dy = center_dy - (next_dy - center_dy)
    fields = (
        (previous_dx, previous_dy),
        (center_dx, center_dy),
        (next_dx, next_dy),
    )
    warped_images = [
        _warp_image_with_inverse_field(image, dx, dy)
        for image, (dx, dy) in zip(images, fields)
    ]
    transformed_backward, transformed_backward_valid = (
        _transform_flow_between_inverse_fields(
            backward_flow,
            backward_valid,
            center_dx,
            center_dy,
            previous_dx,
            previous_dy,
        )
    )
    transformed_forward, transformed_forward_valid = (
        _transform_flow_between_inverse_fields(
            forward_flow,
            forward_valid,
            center_dx,
            center_dy,
            next_dx,
            next_dy,
        )
    )
    return (
        warped_images,
        transformed_backward,
        transformed_backward_valid,
        transformed_forward,
        transformed_forward_valid,
    )


CORRUPTION_FUNCTIONS: dict[str, Callable] = {
    "brightness": image_c.brightness,
    "contrast": image_c.contrast,
    "saturate": image_c.saturate,
    "defocus_blur": image_c.defocus_blur,
    "gaussian_blur": image_c.gaussian_blur,
    "glass_blur": _glass_blur,
    "motion_blur": image_c.motion_blur,
    "zoom_blur": image_c.zoom_blur,
    "gaussian_noise": image_c.gaussian_noise,
    "impulse_noise": _impulse_noise,
    "speckle_noise": image_c.speckle_noise,
    "shot_noise": image_c.shot_noise,
    "pixelate": image_c.pixelate,
    "jpeg_compression": image_c.jpeg_compression,
    "fog": image_c.fog,
    "spatter": image_c.spatter,
}


# RobustSpring frost / rain / snow are content-independent additive layers, corrupted = clip(clean + layer).
# Measured layer correlation t vs t+1: frost 1.00, rain 0.18, snow 0.03 -> frost static, rain/snow per frame.
_OVERLAY_WEATHER = ("frost", "rain", "snow")
_OVERLAY_STATIC = frozenset({"frost"})  # identical layer on every frame
_OVERLAY_DIR = os.environ.get(
    "WEATHER_OVERLAY_DIR", "datasets/weather_overlays"
)
_OVERLAY_CACHE: dict[str, list[np.ndarray]] = {}


def _load_overlays(name: str) -> list[np.ndarray]:
    if name not in _OVERLAY_CACHE:
        paths = sorted(glob.glob(os.path.join(_OVERLAY_DIR, name, "*.png")))
        if not paths:
            raise RuntimeError(
                f"no weather overlays for {name!r} in {_OVERLAY_DIR}; "
                "set WEATHER_OVERLAY_DIR"
            )
        _OVERLAY_CACHE[name] = [
            np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            for path in paths
        ]
    return _OVERLAY_CACHE[name]


def _overlay_weather_frame(image, name, severity, base_seed, frame_index):
    """clip(clean + severity/3 * layer); frost reuses one layer per triplet, rain/snow draw one per frame."""
    overlays = _load_overlays(name)
    static = name in _OVERLAY_STATIC
    sel_seed = (base_seed if static else _frame_seed(base_seed, frame_index)) % (2 ** 32)
    rng = np.random.RandomState(sel_seed)
    layer = overlays[rng.randint(len(overlays))].astype(np.float32)
    if rng.rand() < 0.5:
        layer = layer[:, ::-1, :]  # horizontal flip for extra variety
    img = np.asarray(image, dtype=np.float32)
    if layer.shape != img.shape:
        raise ValueError(f"overlay {name} shape {layer.shape} != image {img.shape}")
    strength = severity / 3.0  # severity 3 reproduces the extracted intensity
    out = np.clip(img + layer * strength, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


class RobustSpringCorruptor:
    """Deterministic corruption of a frame triplet (and its flow labels for elastic)."""

    @staticmethod
    def _apply(image: np.ndarray, name: str, severity: int, seed: int) -> np.ndarray:
        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            np.random.seed(seed)
            random.seed(seed)
            result = CORRUPTION_FUNCTIONS[name](
                Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB"),
                severity=severity,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to apply corruption={name!r}, severity={severity}"
            ) from exc
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

        return np.ascontiguousarray(np.clip(result, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_temporal_sequence(
        images: list[np.ndarray] | tuple[np.ndarray, ...],
        name: str,
        severity: int,
        base_seed: int,
        frame_indices: list[int] | tuple[int, ...] | None = None,
    ) -> list[np.ndarray]:
        """Apply one corruption with deterministic absolute-frame semantics."""
        if frame_indices is None:
            frame_indices = tuple(range(len(images)))
        if len(images) != len(frame_indices):
            raise ValueError("images and frame_indices must have equal length")
        if name in _OVERLAY_WEATHER:
            return [
                _overlay_weather_frame(image, name, severity, base_seed, frame_index)
                for image, frame_index in zip(images, frame_indices)
            ]
        if name == "elastic_transform":
            raise ValueError(
                "elastic_transform needs joint flow-label warping and cannot be "
                "applied as an image-only temporal sequence"
            )
        if name in NOISE_CORRUPTIONS:
            seeds = [
                _frame_seed(base_seed, frame_index)
                for frame_index in frame_indices
            ]
        else:
            seeds = [base_seed] * len(images)
        return [
            RobustSpringCorruptor._apply(image, name, severity, seed)
            for image, seed in zip(images, seeds)
        ]

    @staticmethod
    def apply_triplet(
        images: list[np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray],
        backward_flow: np.ndarray,
        backward_valid: np.ndarray,
        forward_flow: np.ndarray,
        forward_valid: np.ndarray,
        name: str,
        severity: int,
        base_seed: int,
        frame_indices: tuple[int, int, int] = (-1, 0, 1),
    ) -> tuple[
        list[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Apply one corruption to a triplet and both center-origin labels."""
        if name == "elastic_transform":
            return _elastic_triplet(
                images,
                backward_flow,
                backward_valid,
                forward_flow,
                forward_valid,
                severity,
                base_seed,
            )
        outputs = RobustSpringCorruptor.apply_temporal_sequence(
            images,
            name,
            severity,
            base_seed,
            frame_indices=frame_indices,
        )
        return (
            outputs,
            backward_flow,
            backward_valid,
            forward_flow,
            forward_valid,
        )



def impulse_median_gate(images, tau):
    """Inference-time impulse-noise filter: 3x3 median on every frame if any frame has more than
    `tau` percent isolated exact-0/255 pixels (|pixel - local median| > 100).  tau <= 0 disables it."""
    medians = [cv2.medianBlur(image, 3) for image in images]

    def impulse_percent(image, median):
        jump = np.abs(image.astype(np.int16) - median.astype(np.int16)).max(2)
        extreme = ((image == 0) | (image == 255)).any(2)
        return 100.0 * ((jump > 100) & extreme).mean()

    if tau > 0 and max(impulse_percent(i, m) for i, m in zip(images, medians)) > tau:
        return medians, True
    return list(images), False
