import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

import comfy.utils

try:
    import cv2
except Exception:
    cv2 = None


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
    "content_aware": 4,
    "dependency_optimized": 5,
}

BLENDING_METHODS = [
    "gaussian_blur",
    "multi_scale",
    "distance_field",
    "frequency_domain",
    "advanced_feather",
]

SCALING_METHODS = [
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos",
]


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


class ContentAnalyzer:
    """Analyzes image content to optimize tiling strategy."""

    @staticmethod
    def calculate_detail_map(image_tensor: torch.Tensor) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV (cv2) is required for Egregora Analyze Content. "
                "Please install opencv-python to use this node."
            )

        image_np = image_tensor.squeeze(0).cpu().numpy()
        if len(image_np.shape) == 3:
            gray = np.dot(image_np[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = image_np

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        detail_map = cv2.GaussianBlur(gradient_magnitude, (15, 15), 0)
        return detail_map


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


def calculate_enhanced_blur_radius(overlap_x: int, overlap_y: int, blur_scale: float) -> int:
    max_overlap = max(overlap_x, overlap_y)
    if max_overlap <= 0:
        return 0

    min_blur = max(1, int(max_overlap * 0.10))
    max_blur = max(min_blur, int(max_overlap * 0.80))
    blur_scale = max(0.0, min(1.0, blur_scale))
    blur_radius = int(min_blur + (max_blur - min_blur) * blur_scale)
    return max(1, blur_radius)


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


def _partition_length(total: int, parts: int) -> List[int]:
    parts = max(1, parts)
    base = total // parts
    remainder = total % parts
    result = []
    for i in range(parts):
        result.append(base + (1 if i < remainder else 0))
    return result


def _compute_grid(
    up_w: int,
    up_h: int,
    tile_resolution: int,
) -> Tuple[int, int]:
    tile_resolution = max(64, int(tile_resolution))
    grid_x = max(1, int(math.ceil(up_w / float(tile_resolution))))
    grid_y = max(1, int(math.ceil(up_h / float(tile_resolution))))
    return grid_x, grid_y


def _build_tile_boxes(
    image_width: int,
    image_height: int,
    grid_x: int,
    grid_y: int,
    overlap_x: int,
    overlap_y: int,
) -> List[Dict[str, int]]:
    col_widths = _partition_length(image_width, grid_x)
    row_heights = _partition_length(image_height, grid_y)

    xs = [0]
    ys = [0]
    for w in col_widths:
        xs.append(xs[-1] + w)
    for h in row_heights:
        ys.append(ys[-1] + h)

    left_extra = overlap_x // 2
    right_extra = overlap_x - left_extra
    top_extra = overlap_y // 2
    bottom_extra = overlap_y - top_extra

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
                    "row": row,
                    "col": col,
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(max(1, x2 - x1)),
                    "h": int(max(1, y2 - y1)),
                    "base_x": int(base_x1),
                    "base_y": int(base_y1),
                    "base_w": int(max(1, base_x2 - base_x1)),
                    "base_h": int(max(1, base_y2 - base_y1)),
                }
            )
    return boxes


def _tile_display_matrix(
    boxes: List[Dict[str, int]],
    ordered_indices: List[int],
    grid_x: int,
    grid_y: int,
) -> List[List[str]]:
    matrix = [["" for _ in range(grid_x)] for _ in range(grid_y)]
    for order_idx, box_idx in enumerate(ordered_indices, start=1):
        box = boxes[box_idx]
        matrix[box["row"]][box["col"]] = (
            f"{order_idx} ({box['x']},{box['y']}) {box['w']}x{box['h']}"
        )
    return matrix


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
        if row % 2 == 0:
            cols = range(grid_x)
        else:
            cols = range(grid_x - 1, -1, -1)
        for col in cols:
            order.append(row * grid_x + col)
    return order


def _content_aware_order_indices(
    boxes: List[Dict[str, int]],
    detail_map: np.ndarray,
) -> List[int]:
    scored: List[Tuple[int, float]] = []
    h, w = detail_map.shape[:2]
    for idx, box in enumerate(boxes):
        x1 = _clamp_int(box["x"], 0, w)
        y1 = _clamp_int(box["y"], 0, h)
        x2 = _clamp_int(box["x"] + box["w"], 0, w)
        y2 = _clamp_int(box["y"] + box["h"], 0, h)
        if x2 > x1 and y2 > y1:
            score = float(np.mean(detail_map[y1:y2, x1:x2]))
        else:
            score = 0.0
        scored.append((idx, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in scored]


def _dependency_optimized_order_indices(grid_x: int, grid_y: int) -> List[int]:
    scored: List[Tuple[int, int]] = []
    for row in range(grid_y):
        for col in range(grid_x):
            idx = row * grid_x + col
            if (row in (0, grid_y - 1)) and (col in (0, grid_x - 1)):
                priority = 0
            elif row in (0, grid_y - 1) or col in (0, grid_x - 1):
                priority = 1
            else:
                priority = 2
            scored.append((idx, priority))
    scored.sort(key=lambda item: (item[1], item[0]))
    return [idx for idx, _ in scored]


def build_ordered_tile_plan(
    image_width: int,
    image_height: int,
    tile_resolution: int,
    overlap_x: int,
    overlap_y: int,
    grid_x: int,
    grid_y: int,
    tile_order: int,
    detail_map: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, int]], List[List[str]]]:
    boxes = _build_tile_boxes(image_width, image_height, grid_x, grid_y, overlap_x, overlap_y)

    if tile_order == TILE_ORDER_DICT["spiral_outward"]:
        ordered_indices = _spiral_order_indices(grid_x, grid_y, outward=True)
    elif tile_order == TILE_ORDER_DICT["spiral_inward"]:
        ordered_indices = _spiral_order_indices(grid_x, grid_y, outward=False)
    elif tile_order == TILE_ORDER_DICT["serpentine"]:
        ordered_indices = _serpentine_order_indices(grid_x, grid_y)
    elif tile_order == TILE_ORDER_DICT["content_aware"] and detail_map is not None:
        ordered_indices = _content_aware_order_indices(boxes, detail_map)
    elif tile_order == TILE_ORDER_DICT["dependency_optimized"]:
        ordered_indices = _dependency_optimized_order_indices(grid_x, grid_y)
    else:
        ordered_indices = list(range(len(boxes)))

    ordered_boxes: List[Dict[str, int]] = []
    for order_idx, box_idx in enumerate(ordered_indices):
        box = dict(boxes[box_idx])
        proc_w, proc_h = _fit_long_side(box["w"], box["h"], tile_resolution)
        box["order"] = int(order_idx)
        box["source_index"] = int(box_idx)
        box["process_w"] = int(proc_w)
        box["process_h"] = int(proc_h)
        ordered_boxes.append(box)

    matrix = _tile_display_matrix(boxes, ordered_indices, grid_x, grid_y)
    return ordered_boxes, matrix


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


def make_mask_for_box(
    box: Dict[str, int],
    canvas_w: int,
    canvas_h: int,
    feather_size: int,
    blur_scale: float,
    blending_method: str,
) -> np.ndarray:
    tile_w = int(box["w"])
    tile_h = int(box["h"])
    base_x = int(box["base_x"])
    base_y = int(box["base_y"])
    base_w = int(box["base_w"])
    base_h = int(box["base_h"])
    x = int(box["x"])
    y = int(box["y"])

    left_soft = max(0, base_x - x)
    top_soft = max(0, base_y - y)
    right_soft = max(0, (x + tile_w) - (base_x + base_w))
    bottom_soft = max(0, (y + tile_h) - (base_y + base_h))

    feather_size = max(0, int(feather_size))
    left_feather = left_soft if feather_size == 0 else min(left_soft, feather_size)
    right_feather = right_soft if feather_size == 0 else min(right_soft, feather_size)
    top_feather = top_soft if feather_size == 0 else min(top_soft, feather_size)
    bottom_feather = bottom_soft if feather_size == 0 else min(bottom_soft, feather_size)

    yy, xx = np.mgrid[0:tile_h, 0:tile_w]
    mask = np.ones((tile_h, tile_w), dtype=np.float32)

    if left_feather > 0:
        ramp = np.clip(xx / float(max(1, left_feather)), 0.0, 1.0)
        mask *= ramp
    if right_feather > 0:
        dist = (tile_w - 1) - xx
        ramp = np.clip(dist / float(max(1, right_feather)), 0.0, 1.0)
        mask *= ramp
    if top_feather > 0:
        ramp = np.clip(yy / float(max(1, top_feather)), 0.0, 1.0)
        mask *= ramp
    if bottom_feather > 0:
        dist = (tile_h - 1) - yy
        ramp = np.clip(dist / float(max(1, bottom_feather)), 0.0, 1.0)
        mask *= ramp

    if blending_method == "advanced_feather":
        mask = np.power(mask, 0.75)
    elif blending_method == "distance_field":
        mask = np.sqrt(np.clip(mask, 0.0, 1.0))
    elif blending_method == "multi_scale":
        mask = np.power(mask, 1.25)
    elif blending_method == "frequency_domain":
        mask = np.power(mask, 0.90)

    total_soft_x = left_feather + right_feather
    total_soft_y = top_feather + bottom_feather
    blur_radius = calculate_enhanced_blur_radius(total_soft_x, total_soft_y, blur_scale)
    if blur_radius > 0 and (total_soft_x > 0 or total_soft_y > 0):
        pil_mask = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
        if blending_method == "multi_scale":
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=max(1, blur_radius // 2)))
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        else:
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask = np.array(pil_mask, dtype=np.float32) / 255.0

    bx1 = max(0, base_x - x)
    by1 = max(0, base_y - y)
    bx2 = min(tile_w, bx1 + base_w)
    by2 = min(tile_h, by1 + base_h)
    if bx2 > bx1 and by2 > by1:
        mask[by1:by2, bx1:bx2] = 1.0

    return np.clip(mask, 0.0, 1.0)


def _normalize_tiles_input(tiles: Union[List[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
    if isinstance(tiles, list):
        return tiles
    if torch.is_tensor(tiles):
        if tiles.ndim == 4:
            return [tiles[i : i + 1] for i in range(tiles.shape[0])]
        if tiles.ndim == 3:
            return [tiles.unsqueeze(0)]
    raise ValueError("Unsupported tile input format for Egregora Combine.")


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

    RETURN_TYPES = ("IMAGE", "EGREGORA_DATA", "STRING")
    RETURN_NAMES = ("IMAGE", "egregora_data", "ui")
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
        grid_x, grid_y = _compute_grid(up_w, up_h, tile_resolution)

        up = resize_image_tensor(image, up_w, up_h, scaling_method)

        egregora_data = {
            "version": 2,
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
        }

        ordered_boxes, matrix = build_ordered_tile_plan(
            image_width=up_w,
            image_height=up_h,
            tile_resolution=tile_resolution,
            overlap_x=overlap_x,
            overlap_y=overlap_y,
            grid_x=grid_x,
            grid_y=grid_y,
            tile_order=egregora_data["tile_order"],
            detail_map=None,
        )
        egregora_data["tile_boxes"] = ordered_boxes
        egregora_data["tile_matrix"] = matrix

        ui = (
            "Egregora Algorithm\n"
            f"Original: {width}x{height} Upscaled: {up_w}x{up_h}\n"
            f"Grid: {grid_x}x{grid_y} Tile Resolution: {tile_resolution} Overlap: {overlap_x}x{overlap_y}\n"
            f"Order: {tile_order}"
        )
        return up, egregora_data, ui


class Egregora_Analyze_Content:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "egregora_data": ("EGREGORA_DATA",),
            },
        }

    RETURN_TYPES = ("EGREGORA_DATA", "STRING")
    RETURN_NAMES = ("egregora_data", "ui")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"

    def execute(self, image: torch.Tensor, egregora_data: Dict):
        detail_map = ContentAnalyzer.calculate_detail_map(image)
        updated = dict(egregora_data)
        updated["detail_map"] = detail_map

        ordered_boxes, matrix = build_ordered_tile_plan(
            image_width=updated["upscaled_width"],
            image_height=updated["upscaled_height"],
            tile_resolution=updated["tile_resolution"],
            overlap_x=updated["overlap_x"],
            overlap_y=updated["overlap_y"],
            grid_x=updated["grid_x"],
            grid_y=updated["grid_y"],
            tile_order=TILE_ORDER_DICT["content_aware"],
            detail_map=detail_map,
        )
        updated["tile_order"] = TILE_ORDER_DICT["content_aware"]
        updated["tile_boxes"] = ordered_boxes
        updated["tile_matrix"] = matrix

        return updated, "Egregora content analysis completed. Tile order updated to content-aware."


class Egregora_Divide_Select:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "egregora_data": ("EGREGORA_DATA",),
                "tile": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("TILE(S)", "ui")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"

    def execute(self, image: torch.Tensor, egregora_data: Dict, tile: int):
        image_height = image.shape[1]
        image_width = image.shape[2]
        scaling_method = egregora_data.get("scaling_method", "lanczos")

        ordered_boxes = egregora_data.get("tile_boxes")
        if not ordered_boxes:
            ordered_boxes, matrix = build_ordered_tile_plan(
                image_width=image_width,
                image_height=image_height,
                tile_resolution=egregora_data["tile_resolution"],
                overlap_x=egregora_data["overlap_x"],
                overlap_y=egregora_data["overlap_y"],
                grid_x=egregora_data["grid_x"],
                grid_y=egregora_data["grid_y"],
                tile_order=egregora_data["tile_order"],
                detail_map=egregora_data.get("detail_map"),
            )
            egregora_data["tile_boxes"] = ordered_boxes
            egregora_data["tile_matrix"] = matrix

        tile_tensors: List[torch.Tensor] = []
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
            tile_tensors.append(tile_image)

        if not tile_tensors:
            fallback = torch.zeros((1, 64, 64, 3), dtype=image.dtype, device=image.device)
            return [fallback], "No tiles generated"

        matrix = egregora_data.get("tile_matrix")
        if matrix is None:
            matrix = _tile_display_matrix(
                boxes=[dict(box) for box in ordered_boxes],
                ordered_indices=list(range(len(ordered_boxes))),
                grid_x=egregora_data["grid_x"],
                grid_y=egregora_data["grid_y"],
            )

        if tile == 0:
            output_tiles = tile_tensors
        else:
            index = max(0, min(tile - 1, len(tile_tensors) - 1))
            output_tiles = [tile_tensors[index]]

        matrix_ui = "Egregora Tile Matrix:\n" + "\n".join(" ".join(row) for row in matrix)
        return output_tiles, matrix_ui


class Egregora_Combine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "egregora_data": ("EGREGORA_DATA",),
                # Fração do overlap usada para feather por lado.
                # 0.5 = usa metade do overlap em cada borda (máximo recomendado).
                # 0.1 = zona de feather pequena, transição mais abrupta.
                "feather_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.5, "step": 0.01}),
                "scaling_method": (SCALING_METHODS, {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "ui")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"
    INPUT_IS_LIST = True

    @staticmethod
    def _unwrap_scalar(value):
        if isinstance(value, list):
            return value[0] if value else None
        return value

    @staticmethod
    def _normalize_tiles(tiles):
        out = []
        if isinstance(tiles, torch.Tensor):
            if tiles.ndim == 4:
                for i in range(tiles.shape[0]):
                    out.append(tiles[i:i+1])
            elif tiles.ndim == 3:
                out.append(tiles.unsqueeze(0))
            else:
                raise ValueError(f"Unsupported tiles tensor rank: {tiles.ndim}")
            return out
        if isinstance(tiles, (list, tuple)):
            for t in tiles:
                if isinstance(t, torch.Tensor):
                    if t.ndim == 4:
                        for i in range(t.shape[0]):
                            out.append(t[i:i+1])
                    elif t.ndim == 3:
                        out.append(t.unsqueeze(0))
                    else:
                        raise ValueError(f"Unsupported tile rank: {t.ndim}")
                else:
                    raise ValueError(f"Unsupported tile type: {type(t)}")
            return out
        raise ValueError(f"Unsupported tiles input type: {type(tiles)}")

    @staticmethod
    def _build_mask(box, canvas_w, canvas_h, feather_ratio, device):
        w = int(box["w"])
        h = int(box["h"])
        x = int(box["x"])
        y = int(box["y"])
        base_x = int(box.get("base_x", x))
        base_y = int(box.get("base_y", y))
        base_w = int(box.get("base_w", w))
        base_h = int(box.get("base_h", h))

        left_soft   = max(0, base_x - x)
        right_soft  = max(0, (x + w) - (base_x + base_w))
        top_soft    = max(0, base_y - y)
        bottom_soft = max(0, (y + h) - (base_y + base_h))

        feather_w_l = int(left_soft   * feather_ratio)
        feather_w_r = int(right_soft  * feather_ratio)
        feather_h_t = int(top_soft    * feather_ratio)
        feather_h_b = int(bottom_soft * feather_ratio)

        mask = torch.ones((h, w), dtype=torch.float32, device=device)

        if x > 0 and feather_w_l > 0:
            grad = torch.linspace(0.0, 1.0, feather_w_l, device=device)
            mask[:, :feather_w_l] *= grad.unsqueeze(0)

        if (x + w) < canvas_w and feather_w_r > 0:
            grad = torch.linspace(1.0, 0.0, feather_w_r, device=device)
            mask[:, w - feather_w_r:] *= grad.unsqueeze(0)

        if y > 0 and feather_h_t > 0:
            grad = torch.linspace(0.0, 1.0, feather_h_t, device=device)
            mask[:feather_h_t, :] *= grad.unsqueeze(1)

        if (y + h) < canvas_h and feather_h_b > 0:
            grad = torch.linspace(1.0, 0.0, feather_h_b, device=device)
            mask[h - feather_h_b:, :] *= grad.unsqueeze(1)

        return mask

    def execute(self, tiles, egregora_data, feather_ratio, scaling_method):
        tile_list      = self._normalize_tiles(tiles)
        egregora_data  = self._unwrap_scalar(egregora_data)
        feather_ratio  = float(self._unwrap_scalar(feather_ratio))
        scaling_method = self._unwrap_scalar(scaling_method)

        if egregora_data is None:
            raise ValueError("Egregora Combine requires egregora_data.")

        ordered_boxes = egregora_data.get("tile_boxes", [])
        if not ordered_boxes:
            raise ValueError("Egregora Combine requires tile_boxes in egregora_data.")

        canvas_w  = int(egregora_data["upscaled_width"])
        canvas_h  = int(egregora_data["upscaled_height"])
        use_count = min(len(tile_list), len(ordered_boxes))

        if use_count == 0:
            raise ValueError("No tiles available to combine.")

        device = tile_list[0].device
        dtype  = tile_list[0].dtype

        output  = torch.zeros((1, canvas_h, canvas_w, 3), dtype=torch.float32, device=device)
        weights = torch.zeros((1, canvas_h, canvas_w, 1), dtype=torch.float32, device=device)

        for i in range(use_count):
            tile_tensor = tile_list[i].to(dtype=torch.float32)
            box         = ordered_boxes[i]

            if tile_tensor.ndim == 3:
                tile_tensor = tile_tensor.unsqueeze(0)
            if tile_tensor.shape[0] != 1:
                tile_tensor = tile_tensor[:1]

            target_w = int(box["w"])
            target_h = int(box["h"])

            if tile_tensor.shape[2] != target_w or tile_tensor.shape[1] != target_h:
                tile_tensor = resize_image_tensor(tile_tensor, target_w, target_h, scaling_method)

            mask = self._build_mask(
                box=box,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                feather_ratio=feather_ratio,
                device=device,
            )

            mask_4d = mask.unsqueeze(0).unsqueeze(-1)

            x  = int(box["x"])
            y  = int(box["y"])
            x2 = x + target_w
            y2 = y + target_h

            output [:, y:y2, x:x2, :] += tile_tensor * mask_4d
            weights[:, y:y2, x:x2, :] += mask_4d

        weights[weights == 0] = 1.0
        output = output / weights

        final = torch.clamp(output, 0.0, 1.0).to(dtype=dtype)

        overlap_x  = egregora_data.get("overlap_x", 0)
        overlap_y  = egregora_data.get("overlap_y", 0)
        feather_px = int((overlap_x // 2) * feather_ratio)

        ui = (
            "Egregora Combine\n"
            f"Canvas: {canvas_w}x{canvas_h} | Tiles: {use_count}/{len(ordered_boxes)}\n"
            f"Overlap: {overlap_x}x{overlap_y}px | "
            f"Feather ratio: {feather_ratio} (~{feather_px}px/side)"
        )

        return final, ui


NODE_CLASS_MAPPINGS = {
    "Egregora Algorithm": Egregora_Algorithm,
    "Egregora Analyze Content": Egregora_Analyze_Content,
    "Egregora Divide Select": Egregora_Divide_Select,
    "Egregora Combine": Egregora_Combine,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Egregora Algorithm": "Egregora Algorithm",
    "Egregora Analyze Content": "Egregora Analyze Content",
    "Egregora Divide Select": "Egregora Divide Select",
    "Egregora Combine": "Egregora Combine",
}
