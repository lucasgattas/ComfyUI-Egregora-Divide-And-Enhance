
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import comfy.utils



OVERLAP_DICT = {
    "None": 0.0,
    "1/64 Tile": 1.0 / 64.0,
    "1/32 Tile": 1.0 / 32.0,
    "1/16 Tile": 1.0 / 16.0,
    "1/8 Tile": 1.0 / 8.0,
    "1/4 Tile": 1.0 / 4.0,
    "1/2 Tile": 1.0 / 2.0,
    "Adaptive": -1.0,
}

TILE_ORDER_DICT = {
    "linear": 0,
    "spiral_outward": 1,
    "spiral_inward": 2,
    "serpentine": 3,
}

FEATHER_CURVES = ["linear", "smoothstep", "smootherstep", "cosine"]

SCALING_METHODS = [
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos",
]


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))




def calculate_overlap(tile_resolution: int, overlap_fraction: float) -> int:
    return max(0, int(round(tile_resolution * overlap_fraction)))


def calculate_adaptive_overlap_fraction(
    width: int,
    height: int,
    tile_resolution: int,
    target_long_side: int,
) -> float:
    if width <= 0 or height <= 0 or target_long_side <= 0:
        return 0.125

    long_side = max(target_long_side, max(width, height))
    ratio = tile_resolution / float(long_side)

    if ratio >= 0.75:
        return 1.0 / 16.0
    if ratio >= 0.50:
        return 1.0 / 8.0
    if ratio >= 0.33:
        return 1.0 / 6.0
    return 1.0 / 4.0


def _fit_long_side(width: int, height: int, target_long_side: int) -> Tuple[int, int]:
    if width <= 0 or height <= 0:
        return 1, 1

    if target_long_side <= 0:
        return width, height

    long_side = max(width, height)
    if long_side == target_long_side:
        return width, height

    scale = target_long_side / float(long_side)
    fitted_w = max(1, int(round(width * scale)))
    fitted_h = max(1, int(round(height * scale)))
    return fitted_w, fitted_h


def _compute_grid(up_w: int, up_h: int, tile_resolution: int) -> Tuple[int, int]:
    tile_resolution = max(64, int(tile_resolution))
    grid_x = max(1, int(math.ceil(up_w / float(tile_resolution))))
    grid_y = max(1, int(math.ceil(up_h / float(tile_resolution))))
    return grid_x, grid_y


def _spiral_order_indices(grid_x: int, grid_y: int, outward: bool = True) -> List[int]:
    if grid_x <= 0 or grid_y <= 0:
        return []

    cx, cy = grid_x // 2, grid_y // 2
    x, y = cx, cy
    dx, dy = 1, 0
    layer = 1

    visited = set()
    order: List[int] = []

    def append_if_valid(px: int, py: int):
        if 0 <= px < grid_x and 0 <= py < grid_y and (px, py) not in visited:
            visited.add((px, py))
            order.append(py * grid_x + px)

    append_if_valid(x, y)
    while len(order) < grid_x * grid_y:
        for _ in range(2):
            for _ in range(layer):
                x += dx
                y += dy
                append_if_valid(x, y)
            dx, dy = -dy, dx
        layer += 1

    return order if outward else list(reversed(order))


def _serpentine_order_indices(grid_x: int, grid_y: int) -> List[int]:
    order: List[int] = []
    for row in range(grid_y):
        cols = range(grid_x) if row % 2 == 0 else range(grid_x - 1, -1, -1)
        for col in cols:
            order.append(row * grid_x + col)
    return order






def _build_tuki_style_boxes(
    image_width: int,
    image_height: int,
    tile_resolution: int,
    overlap_x: int,
    overlap_y: int,
) -> Tuple[List[Dict[str, int]], int, int]:
    tile_w = min(int(tile_resolution), int(image_width))
    tile_h = min(int(tile_resolution), int(image_height))

    stride_w = max(1, tile_w - int(overlap_x))
    stride_h = max(1, tile_h - int(overlap_y))

    cols = max(1, int(math.ceil((image_width - overlap_x) / float(stride_w))))
    rows = max(1, int(math.ceil((image_height - overlap_y) / float(stride_h))))

    boxes: List[Dict[str, int]] = []
    for r in range(rows):
        for c in range(cols):
            y = r * stride_h
            x = c * stride_w
            if x + tile_w > image_width:
                x = image_width - tile_w
            if y + tile_h > image_height:
                y = image_height - tile_h

            boxes.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "x": int(x),
                    "y": int(y),
                    "w": int(tile_w),
                    "h": int(tile_h),
                    "process_w": int(tile_w),
                    "process_h": int(tile_h),
                }
            )

    return boxes, cols, rows


def build_ordered_tile_plan(
    image_width: int,
    image_height: int,
    tile_resolution: int,
    overlap_x: int,
    overlap_y: int,
    tile_order: int,
) -> Tuple[List[Dict[str, int]], int, int]:
    boxes, grid_x, grid_y = _build_tuki_style_boxes(
        image_width=image_width,
        image_height=image_height,
        tile_resolution=tile_resolution,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
    )

    if tile_order == TILE_ORDER_DICT["spiral_outward"]:
        ordered_indices = _spiral_order_indices(grid_x, grid_y, outward=True)
    elif tile_order == TILE_ORDER_DICT["spiral_inward"]:
        ordered_indices = _spiral_order_indices(grid_x, grid_y, outward=False)
    elif tile_order == TILE_ORDER_DICT["serpentine"]:
        ordered_indices = _serpentine_order_indices(grid_x, grid_y)
    else:
        ordered_indices = list(range(len(boxes)))

    ordered_boxes: List[Dict[str, int]] = []
    for order_idx, box_idx in enumerate(ordered_indices):
        box = dict(boxes[box_idx])
        box["order"] = int(order_idx)
        box["source_index"] = int(box_idx)
        ordered_boxes.append(box)
    return ordered_boxes, grid_x, grid_y


def _image_to_samples(image: torch.Tensor) -> torch.Tensor:
    return image.movedim(-1, 1)


def _samples_to_image(samples: torch.Tensor) -> torch.Tensor:
    return samples.movedim(1, -1)


def resize_image_tensor(
    image: torch.Tensor,
    width: int,
    height: int,
    scaling_method: str = "lanczos",
) -> torch.Tensor:
    samples = _image_to_samples(image)
    resized = comfy.utils.common_upscale(samples, width, height, scaling_method, crop=0)
    return _samples_to_image(resized)


def _apply_feather_curve(grad: torch.Tensor, curve: str) -> torch.Tensor:
    grad = torch.clamp(grad, 0.0, 1.0)
    if curve == "linear":
        return grad
    if curve == "smoothstep":
        return grad * grad * (3.0 - 2.0 * grad)
    if curve == "smootherstep":
        return grad * grad * grad * (grad * (grad * 6.0 - 15.0) + 10.0)
    if curve == "cosine":
        return 0.5 - 0.5 * torch.cos(math.pi * grad)
    return grad


def make_tuki_style_mask(
    x: int,
    y: int,
    tile_w: int,
    tile_h: int,
    canvas_w: int,
    canvas_h: int,
    overlap_x: int,
    overlap_y: int,
    feather_ratio: float,
    feather_curve: str,
    device: torch.device,
) -> torch.Tensor:
    feather_ratio = max(0.0, min(0.5, float(feather_ratio)))

    feather_w = int(max(0, round(overlap_x * feather_ratio)))
    feather_h = int(max(0, round(overlap_y * feather_ratio)))

    mask = torch.ones((tile_h, tile_w), dtype=torch.float32, device=device)

    grad_x = None
    grad_y = None
    if feather_w > 0:
        grad_x = _apply_feather_curve(torch.linspace(0.0, 1.0, feather_w, device=device), feather_curve)
    if feather_h > 0:
        grad_y = _apply_feather_curve(torch.linspace(0.0, 1.0, feather_h, device=device), feather_curve)

    if x > 0 and feather_w > 0:
        mask[:, :feather_w] *= grad_x.unsqueeze(0)
    if x + tile_w < canvas_w and feather_w > 0:
        mask[:, -feather_w:] *= torch.flip(grad_x, dims=[0]).unsqueeze(0)
    if y > 0 and feather_h > 0:
        mask[:feather_h, :] *= grad_y.unsqueeze(1)
    if y + tile_h < canvas_h and feather_h > 0:
        mask[-feather_h:, :] *= torch.flip(grad_y, dims=[0]).unsqueeze(1)

    return mask


def _normalize_tiles(tiles) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if isinstance(tiles, torch.Tensor):
        if tiles.ndim == 4:
            out.extend([tiles[i : i + 1] for i in range(tiles.shape[0])])
        elif tiles.ndim == 3:
            out.append(tiles.unsqueeze(0))
        else:
            raise ValueError(f"Unsupported tiles tensor rank: {tiles.ndim}")
        return out
    if isinstance(tiles, (list, tuple)):
        for t in tiles:
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"Unsupported tile type: {type(t)}")
            if t.ndim == 4:
                out.extend([t[i : i + 1] for i in range(t.shape[0])])
            elif t.ndim == 3:
                out.append(t.unsqueeze(0))
            else:
                raise ValueError(f"Unsupported tile rank: {t.ndim}")
        return out
    raise ValueError(f"Unsupported tiles input type: {type(tiles)}")


def _normalize_masks(masks) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if isinstance(masks, torch.Tensor):
        if masks.ndim == 3:
            out.extend([masks[i] for i in range(masks.shape[0])])
        elif masks.ndim == 2:
            out.append(masks)
        else:
            raise ValueError(f"Unsupported masks tensor rank: {masks.ndim}")
        return out
    if isinstance(masks, (list, tuple)):
        for m in masks:
            if not isinstance(m, torch.Tensor):
                raise ValueError(f"Unsupported mask type: {type(m)}")
            if m.ndim == 3:
                out.extend([m[i] for i in range(m.shape[0])])
            elif m.ndim == 2:
                out.append(m)
            else:
                raise ValueError(f"Unsupported mask rank: {m.ndim}")
        return out
    raise ValueError(f"Unsupported masks input type: {type(masks)}")


class Egregora_Algorithm:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "min_overlap": (list(OVERLAP_DICT.keys()), {"default": "1/8 Tile"}),
                "min_scale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.1}),
                "tile_order": (list(TILE_ORDER_DICT.keys()), {"default": "linear"}),
                "scaling_method": (SCALING_METHODS, {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "EGREGORA_DATA")
    RETURN_NAMES = ("IMAGE", "egregora_data")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"

    def execute(
        self,
        image: torch.Tensor,
        tile_resolution: int,
        min_overlap: str,
        min_scale_factor: float,
        tile_order: str,
        scaling_method: str,
    ):
        _, height, width, _ = image.shape

        min_scale_factor = max(1.0, float(min_scale_factor))
        target_long_side = int(math.ceil(max(width, height) * min_scale_factor))
        up_w, up_h = _fit_long_side(width, height, target_long_side)

        overlap_fraction = OVERLAP_DICT.get(min_overlap, 0.125)
        if overlap_fraction == -1:
            overlap_fraction = calculate_adaptive_overlap_fraction(width, height, tile_resolution, target_long_side)

        overlap_x = calculate_overlap(tile_resolution, overlap_fraction)
        overlap_y = calculate_overlap(tile_resolution, overlap_fraction)

        up = resize_image_tensor(image, up_w, up_h, scaling_method)

        ordered_boxes, grid_x, grid_y = build_ordered_tile_plan(
            image_width=up_w,
            image_height=up_h,
            tile_resolution=tile_resolution,
            overlap_x=overlap_x,
            overlap_y=overlap_y,
            tile_order=TILE_ORDER_DICT.get(tile_order, 0),
        )

        egregora_data = {
            "version": 3,
            "original_width": int(width),
            "original_height": int(height),
            "upscaled_width": int(up_w),
            "upscaled_height": int(up_h),
            "target_long_side": int(target_long_side),
            "tile_resolution": int(tile_resolution),
            "overlap_x": int(overlap_x),
            "overlap_y": int(overlap_y),
            "grid_x": int(grid_x),
            "grid_y": int(grid_y),
            "tile_order": int(TILE_ORDER_DICT.get(tile_order, 0)),
            "scaling_method": scaling_method,
            "tile_boxes": ordered_boxes,
        }

        return up, egregora_data


class Egregora_Divide_Select:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "egregora_data": ("EGREGORA_DATA",),
                "tile": ("INT", {"default": 0, "min": 0, "step": 1}),
                "feather_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.5, "step": 0.01}),
                "feather_curve": (FEATHER_CURVES, {"default": "linear"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("TILE(S)", "MASK(S)")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"

    def execute(
        self,
        image: torch.Tensor,
        egregora_data: Dict,
        tile: int,
        feather_ratio: float,
        feather_curve: str,
    ):
        image_height = image.shape[1]
        image_width = image.shape[2]
        scaling_method = egregora_data.get("scaling_method", "lanczos")
        ordered_boxes = egregora_data.get("tile_boxes", [])
        if not ordered_boxes:
            raise ValueError("Egregora Divide Select requires tile_boxes in egregora_data.")

        overlap_x = int(egregora_data.get("overlap_x", 0))
        overlap_y = int(egregora_data.get("overlap_y", 0))
        canvas_w = int(egregora_data["upscaled_width"])
        canvas_h = int(egregora_data["upscaled_height"])

        tile_tensors: List[torch.Tensor] = []
        mask_tensors: List[torch.Tensor] = []

        for box in ordered_boxes:
            x = _clamp_int(box["x"], 0, image_width - 1)
            y = _clamp_int(box["y"], 0, image_height - 1)
            w = _clamp_int(box["w"], 1, image_width - x)
            h = _clamp_int(box["h"], 1, image_height - y)

            tile_image = image[:, y : y + h, x : x + w, :]
            proc_w = int(box["process_w"])
            proc_h = int(box["process_h"])
            if proc_w != w or proc_h != h:
                tile_image = resize_image_tensor(tile_image, proc_w, proc_h, scaling_method)

            mask = make_tuki_style_mask(
                x=x,
                y=y,
                tile_w=w,
                tile_h=h,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                overlap_x=overlap_x,
                overlap_y=overlap_y,
                feather_ratio=feather_ratio,
                feather_curve=feather_curve,
                device=image.device,
            )

            tile_tensors.append(tile_image)
            mask_tensors.append(mask)

        if not tile_tensors:
            fallback = torch.zeros((1, 64, 64, 3), dtype=image.dtype, device=image.device)
            fallback_mask = torch.ones((64, 64), dtype=torch.float32, device=image.device)
            return [fallback], [fallback_mask]

        if tile == 0:
            return tile_tensors, mask_tensors

        index = max(0, min(tile - 1, len(tile_tensors) - 1))
        return [tile_tensors[index]], [mask_tensors[index]]


class Egregora_Combine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "masks": ("MASK",),
                "egregora_data": ("EGREGORA_DATA",),
                "scaling_method": (SCALING_METHODS, {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"
    INPUT_IS_LIST = True

    @staticmethod
    def _unwrap_scalar(value):
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def execute(self, tiles, masks, egregora_data, scaling_method):
        tile_list = _normalize_tiles(tiles)
        mask_list = _normalize_masks(masks)
        egregora_data = self._unwrap_scalar(egregora_data)
        scaling_method = self._unwrap_scalar(scaling_method)

        if egregora_data is None:
            raise ValueError("Egregora Combine requires egregora_data.")

        ordered_boxes = egregora_data.get("tile_boxes", [])
        if not ordered_boxes:
            raise ValueError("Egregora Combine requires tile_boxes in egregora_data.")

        use_count = min(len(tile_list), len(mask_list), len(ordered_boxes))
        if use_count == 0:
            raise ValueError("No tiles or masks available to combine.")

        canvas_w = int(egregora_data["upscaled_width"])
        canvas_h = int(egregora_data["upscaled_height"])

        device = tile_list[0].device
        in_dtype = tile_list[0].dtype
        channels = tile_list[0].shape[-1] if tile_list[0].ndim == 4 else 3

        output = torch.zeros((1, canvas_h, canvas_w, channels), dtype=torch.float32, device=device)
        weights = torch.zeros((1, canvas_h, canvas_w, 1), dtype=torch.float32, device=device)

        for i in range(use_count):
            tile_tensor = tile_list[i].to(dtype=torch.float32)
            mask_tensor = mask_list[i].to(device=device, dtype=torch.float32)
            box = ordered_boxes[i]

            if tile_tensor.ndim == 3:
                tile_tensor = tile_tensor.unsqueeze(0)
            if tile_tensor.shape[0] != 1:
                tile_tensor = tile_tensor[:1]

            target_w = int(box["w"])
            target_h = int(box["h"])
            if tile_tensor.shape[2] != target_w or tile_tensor.shape[1] != target_h:
                tile_tensor = resize_image_tensor(tile_tensor, target_w, target_h, scaling_method)

            if mask_tensor.ndim != 2:
                raise ValueError("Each mask must be a 2D tensor.")
            if mask_tensor.shape[0] != target_h or mask_tensor.shape[1] != target_w:
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            mask_4d = mask_tensor.unsqueeze(0).unsqueeze(-1)

            x = int(box["x"])
            y = int(box["y"])
            x2 = x + target_w
            y2 = y + target_h

            output[:, y:y2, x:x2, :] += tile_tensor * mask_4d
            weights[:, y:y2, x:x2, :] += mask_4d

        weights = torch.where(weights > 0, weights, torch.ones_like(weights))
        output = output / weights
        final = torch.clamp(output, 0.0, 1.0).to(dtype=in_dtype)
        return (final,)


class Egregora_Debug_Mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "egregora_data": ("EGREGORA_DATA",),
                "feather_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.5, "step": 0.01}),
                "feather_curve": (FEATHER_CURVES, {"default": "linear"}),
                "tile_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "execute"
    CATEGORY = "Egregora/Debug"

    def execute(self, egregora_data, feather_ratio, feather_curve, tile_index):
        if egregora_data is None:
            raise ValueError("Egregora Debug Mask requires egregora_data.")

        ordered_boxes = egregora_data.get("tile_boxes", [])
        if not ordered_boxes:
            raise ValueError("Egregora Debug Mask requires tile_boxes in egregora_data.")

        canvas_w = int(egregora_data["upscaled_width"])
        canvas_h = int(egregora_data["upscaled_height"])
        overlap_x = int(egregora_data.get("overlap_x", 0))
        overlap_y = int(egregora_data.get("overlap_y", 0))

        device = torch.device("cpu")
        masks: List[torch.Tensor] = []

        for box in ordered_boxes:
            masks.append(
                make_tuki_style_mask(
                    x=int(box["x"]),
                    y=int(box["y"]),
                    tile_w=int(box["w"]),
                    tile_h=int(box["h"]),
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    overlap_x=overlap_x,
                    overlap_y=overlap_y,
                    feather_ratio=feather_ratio,
                    feather_curve=feather_curve,
                    device=device,
                )
            )

        tile_index = int(tile_index)
        if tile_index > 0:
            idx = max(0, min(tile_index - 1, len(masks) - 1))
            return (masks[idx].unsqueeze(0),)

        return (torch.stack(masks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Egregora Algorithm": Egregora_Algorithm,
    "Egregora Divide Select": Egregora_Divide_Select,
    "Egregora Combine": Egregora_Combine,
    "Egregora Debug Mask": Egregora_Debug_Mask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Egregora Algorithm": "Egregora Algorithm",
    "Egregora Divide Select": "Egregora Divide Select",
    "Egregora Combine": "Egregora Combine",
    "Egregora Debug Mask": "Egregora Debug Mask",
}
