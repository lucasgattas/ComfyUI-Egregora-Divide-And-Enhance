
import math
from typing import Dict, List, Tuple

import torch
import comfy.utils


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


def _partition_length(total: int, parts: int) -> List[int]:
    parts = max(1, parts)
    base = total // parts
    remainder = total % parts
    return [base + (1 if i < remainder else 0) for i in range(parts)]


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


def _build_owner_padded_boxes(
    image_width: int,
    image_height: int,
    grid_x: int,
    grid_y: int,
    padding_x: int,
    padding_y: int,
) -> List[Dict[str, int]]:
    col_widths = _partition_length(image_width, grid_x)
    row_heights = _partition_length(image_height, grid_y)

    xs = [0]
    ys = [0]
    for w in col_widths:
        xs.append(xs[-1] + w)
    for h in row_heights:
        ys.append(ys[-1] + h)

    left_extra = padding_x // 2
    right_extra = padding_x - left_extra
    top_extra = padding_y // 2
    bottom_extra = padding_y - top_extra

    boxes: List[Dict[str, int]] = []
    for row in range(grid_y):
        for col in range(grid_x):
            base_x1 = xs[col]
            base_x2 = xs[col + 1]
            base_y1 = ys[row]
            base_y2 = ys[row + 1]

            x1 = base_x1 if col == 0 else max(0, base_x1 - left_extra)
            x2 = base_x2 if col == grid_x - 1 else min(image_width, base_x2 + right_extra)
            y1 = base_y1 if row == 0 else max(0, base_y1 - top_extra)
            y2 = base_y2 if row == grid_y - 1 else min(image_height, base_y2 + bottom_extra)

            boxes.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(max(1, x2 - x1)),
                    "h": int(max(1, y2 - y1)),
                    "base_x": int(base_x1),
                    "base_y": int(base_y1),
                    "base_w": int(max(1, base_x2 - base_x1)),
                    "base_h": int(max(1, base_y2 - base_y1)),
                    "process_w": int(max(1, x2 - x1)),
                    "process_h": int(max(1, y2 - y1)),
                }
            )
    return boxes


def build_ordered_tile_plan(
    image_width: int,
    image_height: int,
    tile_resolution: int,
    padding_x: int,
    padding_y: int,
    tile_order: int,
) -> Tuple[List[Dict[str, int]], int, int]:
    grid_x, grid_y = _compute_grid(image_width, image_height, tile_resolution)
    boxes = _build_owner_padded_boxes(
        image_width=image_width,
        image_height=image_height,
        grid_x=grid_x,
        grid_y=grid_y,
        padding_x=padding_x,
        padding_y=padding_y,
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







def _warp_mask(mask: torch.Tensor, strength_px: float, frequency: float, box: Dict[str, int]) -> torch.Tensor:
    strength_px = float(strength_px)
    frequency = max(0.1, float(frequency))
    if strength_px <= 0.0:
        return mask

    import torch.nn.functional as F

    h, w = mask.shape
    if h < 2 or w < 2:
        return mask

    yy = torch.linspace(-1.0, 1.0, h, device=mask.device, dtype=mask.dtype)
    xx = torch.linspace(-1.0, 1.0, w, device=mask.device, dtype=mask.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    row = float(box.get("row", 0))
    col = float(box.get("col", 0))
    idx = float(box.get("source_index", 0))

    fx1 = 1.35 * frequency
    fx2 = 0.85 * frequency
    fy1 = 1.25 * frequency
    fy2 = 0.95 * frequency

    phase_x1 = 0.71 + row * 0.83 + idx * 0.11
    phase_x2 = 1.93 + col * 0.57 + idx * 0.07
    phase_y1 = 1.17 + col * 0.91 + idx * 0.13
    phase_y2 = 2.41 + row * 0.63 + idx * 0.05

    disp_x = (
        0.65 * torch.sin(grid_y * torch.pi * fx1 + phase_x1) +
        0.35 * torch.sin(grid_x * torch.pi * fx2 + phase_x2)
    )
    disp_y = (
        0.65 * torch.sin(grid_x * torch.pi * fy1 + phase_y1) +
        0.35 * torch.sin(grid_y * torch.pi * fy2 + phase_y2)
    )

    band_weight = torch.clamp(5.0 * mask * (1.0 - mask), 0.0, 1.0)
    px_to_norm_x = 2.0 / max(1.0, float(w - 1))
    px_to_norm_y = 2.0 / max(1.0, float(h - 1))

    warped_grid_x = grid_x + disp_x * band_weight * strength_px * px_to_norm_x
    warped_grid_y = grid_y + disp_y * band_weight * strength_px * px_to_norm_y
    grid = torch.stack((warped_grid_x, warped_grid_y), dim=-1)

    x = mask.unsqueeze(0).unsqueeze(0)
    warped = F.grid_sample(
        x,
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.squeeze(0).squeeze(0)



def make_owner_padded_mask(
    box: Dict[str, int],
    blend_x: int,
    blend_y: int,
    feather_curve: str,
    mask_warp_strength: float,
    mask_warp_frequency: float,
    device: torch.device,
) -> torch.Tensor:
    x = int(box["x"])
    y = int(box["y"])
    w = int(box["w"])
    h = int(box["h"])
    base_x = int(box["base_x"])
    base_y = int(box["base_y"])
    base_w = int(box["base_w"])
    base_h = int(box["base_h"])

    left_pad = max(0, base_x - x)
    right_pad = max(0, (x + w) - (base_x + base_w))
    top_pad = max(0, base_y - y)
    bottom_pad = max(0, (y + h) - (base_y + base_h))

    left_blend = min(left_pad, max(0, int(blend_x)))
    right_blend = min(right_pad, max(0, int(blend_x)))
    top_blend = min(top_pad, max(0, int(blend_y)))
    bottom_blend = min(bottom_pad, max(0, int(blend_y)))

    bx1 = max(0, base_x - x)
    by1 = max(0, base_y - y)
    bx2 = min(w, bx1 + base_w)
    by2 = min(h, by1 + base_h)

    yy = torch.arange(h, device=device, dtype=torch.float32)
    xx = torch.arange(w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    # Gentle, deterministic irregular contour for the core boundary.
    # IMPORTANT: only sides that actually blend are allowed to move.
    strength = max(0.0, float(mask_warp_strength))
    frequency = max(0.1, float(mask_warp_frequency))

    row = float(box.get("row", 0))
    col = float(box.get("col", 0))
    idx = float(box.get("source_index", 0))

    amp_left = min(float(left_blend), strength) if left_blend > 0 else 0.0
    amp_right = min(float(right_blend), strength) if right_blend > 0 else 0.0
    amp_top = min(float(top_blend), strength) if top_blend > 0 else 0.0
    amp_bottom = min(float(bottom_blend), strength) if bottom_blend > 0 else 0.0

    t_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32)
    t_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32)

    if amp_left > 0.0:
        left_offset = amp_left * (
            0.65 * torch.sin(t_y * math.pi * (1.20 * frequency) + (0.73 + row * 0.81 + idx * 0.13)) +
            0.35 * torch.sin(t_y * math.pi * (2.05 * frequency) + (1.91 + col * 0.37 + idx * 0.07))
        )
    else:
        left_offset = torch.zeros((h,), device=device, dtype=torch.float32)

    if amp_right > 0.0:
        right_offset = amp_right * (
            0.65 * torch.sin(t_y * math.pi * (1.17 * frequency) + (1.33 + row * 0.57 + idx * 0.09)) +
            0.35 * torch.sin(t_y * math.pi * (1.93 * frequency) + (2.17 + col * 0.41 + idx * 0.05))
        )
    else:
        right_offset = torch.zeros((h,), device=device, dtype=torch.float32)

    if amp_top > 0.0:
        top_offset = amp_top * (
            0.65 * torch.sin(t_x * math.pi * (1.14 * frequency) + (1.11 + col * 0.79 + idx * 0.12)) +
            0.35 * torch.sin(t_x * math.pi * (2.11 * frequency) + (2.43 + row * 0.29 + idx * 0.06))
        )
    else:
        top_offset = torch.zeros((w,), device=device, dtype=torch.float32)

    if amp_bottom > 0.0:
        bottom_offset = amp_bottom * (
            0.65 * torch.sin(t_x * math.pi * (1.28 * frequency) + (0.89 + col * 0.61 + idx * 0.08)) +
            0.35 * torch.sin(t_x * math.pi * (1.87 * frequency) + (2.71 + row * 0.33 + idx * 0.04))
        )
    else:
        bottom_offset = torch.zeros((w,), device=device, dtype=torch.float32)

    left_edge = bx1 + left_offset.unsqueeze(1)
    right_edge = bx2 + right_offset.unsqueeze(1)
    top_edge = by1 + top_offset.unsqueeze(0)
    bottom_edge = by2 + bottom_offset.unsqueeze(0)

    # 1D weights with the irregular core as the full-ownership plateau.
    if left_blend > 0:
        t = (grid_x - (left_edge - left_blend)) / max(1.0, float(left_blend))
        w_left = _apply_feather_curve(torch.clamp(t, 0.0, 1.0), feather_curve)
    else:
        w_left = torch.ones_like(grid_x)

    if right_blend > 0:
        t = ((right_edge + right_blend) - grid_x) / max(1.0, float(right_blend))
        w_right = _apply_feather_curve(torch.clamp(t, 0.0, 1.0), feather_curve)
    else:
        w_right = torch.ones_like(grid_x)

    if top_blend > 0:
        t = (grid_y - (top_edge - top_blend)) / max(1.0, float(top_blend))
        w_top = _apply_feather_curve(torch.clamp(t, 0.0, 1.0), feather_curve)
    else:
        w_top = torch.ones_like(grid_y)

    if bottom_blend > 0:
        t = ((bottom_edge + bottom_blend) - grid_y) / max(1.0, float(bottom_blend))
        w_bottom = _apply_feather_curve(torch.clamp(t, 0.0, 1.0), feather_curve)
    else:
        w_bottom = torch.ones_like(grid_y)

    mask = w_left * w_right * w_top * w_bottom
    mask = torch.clamp(mask, 0.0, 1.0)

    # Only warp where there is an actual transition band; keep outer canvas edges solid.
    if mask_warp_strength > 0.0 and (left_blend > 0 or right_blend > 0 or top_blend > 0 or bottom_blend > 0):
        mask = _warp_mask(mask, mask_warp_strength, mask_warp_frequency, box)
        mask = torch.clamp(mask, 0.0, 1.0)

    # Guarantee solid ownership on outer canvas borders.
    if left_blend == 0:
        mask[:, :bx1+1] = 1.0
    if right_blend == 0:
        mask[:, bx2-1:] = 1.0
    if top_blend == 0:
        mask[:by1+1, :] = 1.0
    if bottom_blend == 0:
        mask[by2-1:, :] = 1.0

    return torch.clamp(mask, 0.0, 1.0)


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
                "padding_px": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 1}),
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
        padding_px: int,
        min_scale_factor: float,
        tile_order: str,
        scaling_method: str,
    ):
        _, height, width, _ = image.shape

        min_scale_factor = max(1.0, float(min_scale_factor))
        target_long_side = int(math.ceil(max(width, height) * min_scale_factor))
        up_w, up_h = _fit_long_side(width, height, target_long_side)

        padding_px = max(0, int(padding_px))
        padding_x = padding_px
        padding_y = padding_px

        up = resize_image_tensor(image, up_w, up_h, scaling_method)

        ordered_boxes, grid_x, grid_y = build_ordered_tile_plan(
            image_width=up_w,
            image_height=up_h,
            tile_resolution=tile_resolution,
            padding_x=padding_x,
            padding_y=padding_y,
            tile_order=TILE_ORDER_DICT.get(tile_order, 0),
        )

        egregora_data = {
            "version": 5,
            "original_width": int(width),
            "original_height": int(height),
            "upscaled_width": int(up_w),
            "upscaled_height": int(up_h),
            "target_long_side": int(target_long_side),
            "tile_resolution": int(tile_resolution),
            "padding_x": int(padding_x),
            "padding_y": int(padding_y),
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
                "blend_px": ("INT", {"default": 48, "min": 0, "max": 1024, "step": 1}),
                "feather_curve": (FEATHER_CURVES, {"default": "smootherstep"}),
                "mask_warp_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.1}),
                "mask_warp_frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
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
        blend_px: int,
        feather_curve: str,
        mask_warp_strength: float,
        mask_warp_frequency: float,
    ):
        image_height = image.shape[1]
        image_width = image.shape[2]
        scaling_method = egregora_data.get("scaling_method", "lanczos")
        ordered_boxes = egregora_data.get("tile_boxes", [])
        if not ordered_boxes:
            raise ValueError("Egregora Divide Select requires tile_boxes in egregora_data.")

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

            mask = make_owner_padded_mask(
                box=box,
                blend_x=blend_px,
                blend_y=blend_px,
                feather_curve=feather_curve,
                mask_warp_strength=mask_warp_strength,
                mask_warp_frequency=mask_warp_frequency,
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

            mask_4d = torch.clamp(mask_tensor, 0.0, 1.0).unsqueeze(0).unsqueeze(-1)

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
                "blend_px": ("INT", {"default": 48, "min": 0, "max": 1024, "step": 1}),
                "feather_curve": (FEATHER_CURVES, {"default": "smootherstep"}),
                "mask_warp_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.1}),
                "mask_warp_frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "tile_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "execute"
    CATEGORY = "Egregora/Debug"

    def execute(self, egregora_data, blend_px, feather_curve, mask_warp_strength, mask_warp_frequency, tile_index):
        if egregora_data is None:
            raise ValueError("Egregora Debug Mask requires egregora_data.")

        ordered_boxes = egregora_data.get("tile_boxes", [])
        if not ordered_boxes:
            raise ValueError("Egregora Debug Mask requires tile_boxes in egregora_data.")

        device = torch.device("cpu")
        masks = [
            make_owner_padded_mask(
                box=box,
                blend_x=blend_px,
                blend_y=blend_px,
                feather_curve=feather_curve,
                mask_warp_strength=mask_warp_strength,
                mask_warp_frequency=mask_warp_frequency,
                device=device,
            )
            for box in ordered_boxes
        ]

        if not masks:
            return (torch.ones((1, 64, 64), dtype=torch.float32),)

        tile_index = int(tile_index)
        if tile_index <= 0:
            return (torch.stack(masks, dim=0),)

        idx = max(0, min(tile_index - 1, len(masks) - 1))
        return (masks[idx].unsqueeze(0),)


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
