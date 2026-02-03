import sys
import os
import torch
import math
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import comfy.utils
from comfy import model_management
try:
    import cv2
except Exception:
    cv2 = None
from typing import Tuple, List, Dict, Optional

# Constants and helper functions from the original file for reference
OVERLAP_DICT = {
    "None": 0, "1/64 Tile": 0.015625, "1/32 Tile": 0.03125,
    "1/16 Tile": 0.0625, "1/8 Tile": 0.125, "1/4 Tile": 0.25,
    "1/2 Tile": 0.5, "Adaptive": -1,
}

TILE_ORDER_DICT = {
    "linear": 0, "spiral_outward": 1, "spiral_inward": 2,
    "serpentine": 3, "content_aware": 4, "dependency_optimized": 5
}

BLENDING_METHODS = [
    "gaussian_blur", "multi_scale", "distance_field",
    "frequency_domain", "advanced_feather"
]

SCALING_METHODS = [
    "nearest-exact", "bilinear", "area", "bicubic", "lanczos"
]

class ContentAnalyzer:
    """Analyzes image content to optimize tiling strategy"""
    
    @staticmethod
    def calculate_detail_map(image_tensor: torch.Tensor) -> np.ndarray:
        """Calculate detail/complexity map of image using gradient magnitude"""
        if cv2 is None:
            raise RuntimeError(
                "OpenCV (cv2) is required for Egregora Analyze Content. "
                "Please install opencv-python to use this node."
            )
        image_np = image_tensor.squeeze(0).cpu().numpy()
        if len(image_np.shape) == 3:
            gray = np.dot(image_np, [0.299, 0.587, 0.114])
        else:
            gray = image_np
            
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        detail_map = cv2.GaussianBlur(gradient_magnitude, (15, 15), 0)
        
        return detail_map

def calculate_overlap(tile_size, overlap_fraction):
    """Calculate overlap with bounds checking"""
    return max(0, int(overlap_fraction * tile_size))

def calculate_adaptive_overlap_fraction(W, H, tw, th):
    """
    Adaptive overlap heuristic:
    - Larger tiles relative to the image need less overlap.
    - Smaller tiles need more overlap to hide seams.
    """
    if W <= 0 or H <= 0:
        return 0.125

    ratio_w = tw / float(W)
    ratio_h = th / float(H)
    ratio = min(ratio_w, ratio_h)

    if ratio >= 0.75:
        return 1.0 / 16.0  # 0.0625
    if ratio >= 0.50:
        return 1.0 / 8.0   # 0.125
    if ratio >= 0.33:
        return 1.0 / 6.0   # ~0.1667
    return 1.0 / 4.0       # 0.25

def calculate_enhanced_blur_radius(overlap_x, overlap_y, blur_scale):
    """FIXED: Calculate blur radius with proper scaling - higher blur_scale = MORE blur"""
    max_overlap = max(overlap_x, overlap_y)
    if max_overlap <= 0:
        return 0
    
    # Fixed logic: higher blur_scale should create MORE blur
    min_blur = max(1, int(max_overlap * 0.1))  # Minimum 10% of overlap
    max_blur = max(min_blur, int(max_overlap * 0.8))  # Maximum 80% of overlap
    
    # Linear interpolation: 0.0 = min_blur, 1.0 = max_blur
    blur_scale = max(0.0, min(1.0, blur_scale))
    blur_radius = int(min_blur + (max_blur - min_blur) * blur_scale)
    
    return max(1, blur_radius)

def create_enhanced_tile_coordinates(image_width: int, image_height: int, 
                                   tile_width: int, tile_height: int,
                                   overlap_x: int, overlap_y: int,
                                   grid_x: int, grid_y: int,
                                   tile_order: int,
                                   detail_map: Optional[np.ndarray] = None) -> Tuple[List[Tuple[int, int]], List[List[str]]]:
    """Enhanced tile coordinate generation with multiple ordering strategies"""
    
    tiles = []
    matrix = [['' for _ in range(grid_x)] for _ in range(grid_y)]
    
    for row in range(grid_y):
        y = row * (tile_height - overlap_y)
        if row == grid_y - 1:
            y = image_height - tile_height
        for col in range(grid_x):
            x = col * (tile_width - overlap_x)
            if col == grid_x - 1:
                x = image_width - tile_width
            tiles.append((max(0, x), max(0, y)))
    
    # Apply different ordering strategies
    if tile_order == 1:  # Spiral outward
        tiles = _spiral_order(tiles, grid_x, grid_y, outward=True)
    elif tile_order == 2:  # Spiral inward
        tiles = _spiral_order(tiles, grid_x, grid_y, outward=False)
    elif tile_order == 3:  # Serpentine
        tiles = _serpentine_order(tiles, grid_x, grid_y)
    elif tile_order == 4:  # Content aware
        if detail_map is not None:
            tiles = _content_aware_order(tiles, grid_x, grid_y, detail_map, tile_width, tile_height)
    elif tile_order == 5:  # Dependency optimized
        tiles = _dependency_optimized_order(tiles, grid_x, grid_y)
    
    # Rebuild matrix for display
    for i, (x, y) in enumerate(tiles):
        row = min(grid_y - 1, max(0, y // max(1, tile_height - overlap_y)))
        col = min(grid_x - 1, max(0, x // max(1, tile_width - overlap_x)))
        matrix[row][col] = f"{i + 1} ({x},{y})"
    
    return tiles, matrix

def _spiral_order(tiles: List[Tuple[int, int]], grid_x: int, grid_y: int, outward: bool = True) -> List[Tuple[int, int]]:
    """Generate spiral tile order"""
    if not tiles:
        return tiles
        
    spiral_tiles = []
    visited = set()
    
    cx, cy = grid_x // 2, grid_y // 2
    x, y = cx, cy
    dx, dy = 1, 0
    layer = 1
    
    if 0 <= x < grid_x and 0 <= y < grid_y:
        idx = y * grid_x + x
        if idx < len(tiles):
            spiral_tiles.append(tiles[idx])
            visited.add((x, y))
    
    while len(spiral_tiles) < len(tiles):
        for _ in range(2):
            for _ in range(layer):
                if 0 <= x < grid_x and 0 <= y < grid_y and (x, y) not in visited:
                    idx = y * grid_x + x
                    if idx < len(tiles):
                        spiral_tiles.append(tiles[idx])
                        visited.add((x, y))
                x += dx
                y += dy
            dx, dy = -dy, dx
        layer += 1
    
    return spiral_tiles if outward else spiral_tiles[::-1]

def _serpentine_order(tiles: List[Tuple[int, int]], grid_x: int, grid_y: int) -> List[Tuple[int, int]]:
    """Generate serpentine (snake-like) tile order"""
    serpentine_tiles = []
    
    for row in range(grid_y):
        if row % 2 == 0:
            for col in range(grid_x):
                idx = row * grid_x + col
                if idx < len(tiles):
                    serpentine_tiles.append(tiles[idx])
        else:
            for col in range(grid_x - 1, -1, -1):
                idx = row * grid_x + col
                if idx < len(tiles):
                    serpentine_tiles.append(tiles[idx])
    
    return serpentine_tiles

def _content_aware_order(tiles: List[Tuple[int, int]], grid_x: int, grid_y: int,
                        detail_map: np.ndarray, tile_width: int, tile_height: int) -> List[Tuple[int, int]]:
    """Order tiles by content complexity (high detail first)"""
    
    def get_tile_complexity(tile_coord):
        x, y = tile_coord
        h, w = detail_map.shape
        
        x1, x2 = max(0, x), min(w, x + tile_width)
        y1, y2 = max(0, y), min(h, y + tile_height)
        
        if x2 > x1 and y2 > y1:
            return np.mean(detail_map[y1:y2, x1:x2])
        return 0.0
    
    tiles_with_complexity = [(tile, get_tile_complexity(tile)) for tile in tiles]
    tiles_with_complexity.sort(key=lambda x: x[1], reverse=True)
    
    return [tile for tile, _ in tiles_with_complexity]

def _dependency_optimized_order(tiles: List[Tuple[int, int]], grid_x: int, grid_y: int) -> List[Tuple[int, int]]:
    """Order tiles to minimize dependencies"""
    
    def get_tile_priority(idx):
        row, col = idx // grid_x, idx % grid_x
        
        if (row == 0 or row == grid_y - 1) and (col == 0 or col == grid_x - 1):
            return 0
        elif row == 0 or row == grid_y - 1 or col == 0 or col == grid_x - 1:
            return 1
        else:
            return 2
    
    indexed_tiles = [(i, tiles[i], get_tile_priority(i)) for i in range(len(tiles))]
    indexed_tiles.sort(key=lambda x: (x[2], x[0]))
    
    return [tile for _, tile, _ in indexed_tiles]

class Egregora_Algorithm:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "tile_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "min_overlap": (list(OVERLAP_DICT.keys()), {"default": "1/8 Tile"}),
                "min_scale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "tile_order": (list(TILE_ORDER_DICT.keys()), {"default": "linear"}),
                "scaling_method": (SCALING_METHODS, {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "EGREGORA_DATA", "STRING")
    RETURN_NAMES = ("IMAGE", "egregora_data", "ui")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"

    def _calc_grid(self, W, H, tw, th, ox, oy, min_sf):
        import math
        # Overlap-flexible, minimal-tiles strategy:
        # - Never increase overlap above user request.
        # - Compute the smallest grid that meets min scale.
        # - Reduce overlap if needed to avoid extra tiles.
        min_sf = max(min_sf, 1.0)

        target_w = int(math.ceil(W * min_sf))
        target_h = int(math.ceil(H * min_sf))

        # Grid counts: minimal number of tiles to cover target size
        gx = max(1, int(math.ceil(target_w / tw)))
        gy = max(1, int(math.ceil(target_h / th)))

        # Max overlap that still covers target size with this grid
        if gx <= 1:
            ox_eff = 0
        else:
            max_ox = (tw * gx - target_w) / float(gx - 1)
            ox_eff = int(math.floor(max_ox))
            ox_eff = max(0, min(ox_eff, ox))

        if gy <= 1:
            oy_eff = 0
        else:
            max_oy = (th * gy - target_h) / float(gy - 1)
            oy_eff = int(math.floor(max_oy))
            oy_eff = max(0, min(oy_eff, oy))

        upW = int(tw * gx - ox_eff * (gx - 1))
        upH = int(th * gy - oy_eff * (gy - 1))

        # Safety clamps
        upW = max(tw, upW)
        upH = max(th, upH)
        ox_eff = max(0, min(ox_eff, tw - 1))
        oy_eff = max(0, min(oy_eff, th - 1))

        return upW, upH, gx, gy, ox_eff, oy_eff

    def execute(self, image, tile_width, tile_height, min_overlap, min_scale_factor,
                tile_order, scaling_method):

        import comfy
        _, H, W, _ = image.shape
        ov = OVERLAP_DICT.get(min_overlap, 0.125)
        if ov == -1:
            ov = calculate_adaptive_overlap_fraction(W, H, tile_width, tile_height)
        ox = calculate_overlap(tile_width, ov)
        oy = calculate_overlap(tile_height, ov)

        upW, upH, gx, gy, ox, oy = self._calc_grid(W, H, tile_width, tile_height, ox, oy, min_scale_factor)

        samples = image.movedim(-1, 1)
        up = comfy.utils.common_upscale(samples, upW, upH, scaling_method, crop=0).movedim(1, -1)

        egregora_data = {
            "upscaled_width": int(upW), "upscaled_height": int(upH),
            "tile_width": int(tile_width), "tile_height": int(tile_height),
            "overlap_x": int(ox), "overlap_y": int(oy),
            "grid_x": int(gx), "grid_y": int(gy),
            "tile_order": int(TILE_ORDER_DICT.get(tile_order, 0)),
        }

        ui = (f"Egregora Algorithm\n"
              f"Original: {W}x{H}  Upscaled: {upW}x{upH}\n"
              f"Grid: {gx}x{gy}  Tile: {tile_width}x{tile_height}  Overlap: {ox}x{oy}\n"
              f"Order: {tile_order}")
        return (up, egregora_data, ui)

class Egregora_Divide_Select:
    @classmethod
    def INPUT_TYPES(s):
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

    def execute(self, image, egregora_data, tile):
        image_height = image.shape[1]
        image_width = image.shape[2]

        tile_width = egregora_data['tile_width']
        tile_height = egregora_data['tile_height']
        overlap_x = egregora_data['overlap_x']
        overlap_y = egregora_data['overlap_y']
        grid_x = egregora_data['grid_x']
        grid_y = egregora_data['grid_y']
        tile_order = egregora_data['tile_order']
        detail_map = egregora_data.get('detail_map')

        tile_coordinates, matrix = create_enhanced_tile_coordinates(
            image_width, image_height, tile_width, tile_height, 
            overlap_x, overlap_y, grid_x, grid_y, tile_order, detail_map
        )

        image_tiles = []
        for tile_coord in tile_coordinates:
            x, y = tile_coord
            
            x = max(0, min(x, image_width - tile_width))
            y = max(0, min(y, image_height - tile_height))
            
            image_tile = image[
                :,
                y : y + tile_height,
                x : x + tile_width,
                :,
            ]
            image_tiles.append(image_tile)

        if not image_tiles:
            return ([torch.zeros((1, tile_height, tile_width, 3))], "No tiles generated")

        all_tiles = torch.cat(image_tiles, dim=0)

        if tile == 0:
            tile_or_tiles = all_tiles
        else:
            if tile <= len(image_tiles):
                tile_or_tiles = image_tiles[tile - 1]
            else:
                tile_or_tiles = image_tiles[0]

        matrix_ui = "Egregora Tile Matrix:\n" + '\n'.join([' '.join(row) for row in matrix])

        return ([tile_or_tiles[i].unsqueeze(0) for i in range(tile_or_tiles.shape[0])], matrix_ui)

import math, torch, numpy as np
from PIL import Image, ImageDraw, ImageFilter

import math
import torch
import numpy as np
from typing import List, Tuple


class Egregora_Combine:
    """
    Egregora-order + DaC-style alpha-over with controllable edge-feather transparency.

    - Coordinates: uses create_enhanced_tile_coordinates (same as Divide & Select),
      so order/positions match exactly.
    - Per-tile analytic feather mask that goes to 0 (transparent) right at
      tile borders, rising to 1 (opaque) over "feather_size" pixels.
      Feather is applied ONLY on internal sides (not on outer canvas edges),
      and clamped to the tile overlap so it cannot create holes.
    - Compositing: ORDER-DEPENDENT alpha-over (no global normalization), like DaC.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "egregora_data": ("EGREGORA_DATA",),
            },
            "optional": {
                "feather_size": ("INT", {"default": 16, "min": 0, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "ui")
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"

    # ---- coordinate helper: call Egregora's own generator ----
    def _coords_egregora(self, W, H, tw, th, ox, oy, gx, gy, order, detail_map=None):
        coords, _ = create_enhanced_tile_coordinates(
            W, H, tw, th, ox, oy, gx, gy, order, detail_map
        )
        return coords

    @staticmethod
    def _to_int_scalar(value, default=16):
        """Unwrap lists/tuples/tensors and cast to int safely."""
        import torch
        v = value
        if isinstance(v, (list, tuple)):
            v = v[0] if len(v) else default
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().flatten().tolist()
            v = v[0] if len(v) else default
        try:
            v = int(float(v))
        except Exception:
            v = default
        return v

    @staticmethod
    def _build_feather_mask(H, W, device, dtype,
                            xs, ys, up_w, up_h,
                            feather_size, overlap_x, overlap_y):
        """
        Analytic feather mask that is 0 at the ROI edges and rises linearly to 1
        over 'feather_size' pixels, ONLY on sides that don't touch the canvas border.
        The feather per-axis is clamped to the corresponding overlap to prevent gaps.
        """
        fx = max(0, min(int(feather_size), int(overlap_x), max(0, W - 1)))
        fy = max(0, min(int(feather_size), int(overlap_y), max(0, H - 1)))

        at_left   = (xs == 0)
        at_top    = (ys == 0)
        at_right  = (xs + W == up_w)
        at_bottom = (ys + H == up_h)

        fl = 0 if at_left   else fx
        fr = 0 if at_right  else fx
        ft = 0 if at_top    else fy
        fb = 0 if at_bottom else fy

        y = torch.arange(H, device=device, dtype=dtype)[:, None]  # (H,1)
        x = torch.arange(W, device=device, dtype=dtype)[None, :]  # (1,W)

        dist_left   = x
        dist_right  = (W - 1) - x
        dist_top    = y
        dist_bottom = (H - 1) - y

        one = torch.ones((H, W), device=device, dtype=dtype)

        def ramp(dist, f):
            if f <= 0:
                return one
            return torch.clamp(dist / float(f), 0.0, 1.0)

        mask_left   = ramp(dist_left,   fl)
        mask_right  = ramp(dist_right,  fr)
        mask_top    = ramp(dist_top,    ft)
        mask_bottom = ramp(dist_bottom, fb)

        mask2d = torch.minimum(torch.minimum(mask_left, mask_right),
                               torch.minimum(mask_top,  mask_bottom))
        return mask2d.unsqueeze(0).unsqueeze(-1)  # (1,H,W,1)

    def execute(self, images, egregora_data, feather_size=16):
        # Flatten tiles preserving order
        flat = []
        for itm in images:
            if isinstance(itm, (list, tuple)):
                flat.extend(itm)
            else:
                flat.append(itm)
        tiles = torch.cat([t if t.dim() == 4 else t.unsqueeze(0) for t in flat], dim=0)

        # egregora_data may arrive wrapped (because INPUT_IS_LIST=True)
        if isinstance(egregora_data, (list, tuple)):
            egregora_data = egregora_data[0] if len(egregora_data) else {}

        # NEW: unwrap feather_size safely (it may arrive as [16] or a tensor)
        feather_size = self._to_int_scalar(feather_size, default=16)
        if feather_size < 0:
            feather_size = 0

        # Geometry from the algorithm node
        up_w = int(egregora_data["upscaled_width"])
        up_h = int(egregora_data["upscaled_height"])
        tw   = int(egregora_data["tile_width"])
        th   = int(egregora_data["tile_height"])
        ox   = int(egregora_data["overlap_x"])
        oy   = int(egregora_data["overlap_y"])
        gx   = int(egregora_data["grid_x"])
        gy   = int(egregora_data["grid_y"])
        order = int(egregora_data.get("tile_order", 0))
        detail_map = egregora_data.get("detail_map", None)

        # Clamp feather to avoid gaps with alpha-over compositing
        # (effective <= overlap/2 on the tighter axis)
        max_feather = max(0, min(ox, oy) // 2)
        feather_size = min(feather_size, max_feather)

        # Coordinates using same generator/order as Divide & Select
        coords = self._coords_egregora(up_w, up_h, tw, th, ox, oy, gx, gy, order, detail_map)

        # Canvas
        device, dtype = tiles.device, tiles.dtype
        canvas = torch.zeros((1, up_h, up_w, 3), dtype=dtype, device=device)

        # Composite in that same order
        T = tiles.shape[0]
        for i, (x, y) in enumerate(coords):
            if i >= T:
                break

            tile = tiles[i].squeeze(0).clamp(0, 1)  # (H,W,3)
            H, W = int(tile.shape[0]), int(tile.shape[1])

            xs = max(0, int(x)); ys = max(0, int(y))
            xe = min(xs + W, up_w); ye = min(ys + H, up_h)
            if xe <= xs or ye <= ys:
                continue
            Wroi, Hroi = (xe - xs), (ye - ys)

            tile_roi = tile[:Hroi, :Wroi, :]

            mask = self._build_feather_mask(
                H=Hroi, W=Wroi, device=device, dtype=dtype,
                xs=xs, ys=ys, up_w=up_w, up_h=up_h,
                feather_size=feather_size, overlap_x=ox, overlap_y=oy
            )  # (1,Hroi,Wroi,1)

            roi = canvas[:, ys:ye, xs:xe, :]
            canvas[:, ys:ye, xs:xe, :] = roi * (1.0 - mask) + tile_roi.unsqueeze(0) * mask

        eff_fx = max(0, min(int(feather_size), ox))
        eff_fy = max(0, min(int(feather_size), oy))
        ui = (f"Egregora-order + DaC-style Combine\n"
              f"Canvas: {up_w}x{up_h}  Grid: {gx}x{gy} ({gx*gy} tiles)\n"
              f"Overlap: {ox}x{oy}  Order: {order}\n"
              f"Feather: effective={eff_fx}x{eff_fy}px (clamped <= overlap/2)\n"
              f"(internal sides only; clamped to overlap)")
        return (canvas.clamp(0, 1), ui)


class Egregora_Preview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "egregora_data": ("EGREGORA_DATA",),
                "show_grid": ("BOOLEAN", {"default": True}),
                "show_overlap": ("BOOLEAN", {"default": True}),
                "show_order": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "feather_size": ("INT", {"default": 32, "min": 0, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "ui")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Utils"
    DESCRIPTION = "Preview the tiling strategy with visual indicators for debugging seam issues."

    def execute(self, image, egregora_data, show_grid, show_overlap, show_order, feather_size=32):
        _, height, width, _ = image.shape
        
        tile_width = egregora_data['tile_width']
        tile_height = egregora_data['tile_height']
        overlap_x = egregora_data['overlap_x']
        overlap_y = egregora_data['overlap_y']
        grid_x = egregora_data['grid_x']
        grid_y = egregora_data['grid_y']
        tile_order = egregora_data['tile_order']
        detail_map = egregora_data.get('detail_map')

        tile_coordinates, matrix = create_enhanced_tile_coordinates(
            width, height, tile_width, tile_height,
            overlap_x, overlap_y, grid_x, grid_y, tile_order, detail_map
        )

        preview = image.clone()
        preview_np = (preview.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        preview_pil = Image.fromarray(preview_np)
        draw = ImageDraw.Draw(preview_pil, 'RGBA')

        # Enhanced colors for better visibility
        grid_color = (255, 0, 0, 180)      # Bright red for tile boundaries
        overlap_color = (0, 255, 0, 100)   # Semi-transparent green for overlaps
        order_color = (255, 255, 0, 255)   # Bright yellow for numbers
        feather_color = (0, 0, 255, 80)    # Blue for feather zones

        if isinstance(feather_size, (list, tuple)):
            feather_size = feather_size[0] if len(feather_size) else 32
        try:
            feather_size = int(feather_size)
        except Exception:
            feather_size = 32

        for idx, (x, y) in enumerate(tile_coordinates):
            if show_grid:
                # Draw tile boundary with thicker lines
                draw.rectangle([x, y, x + tile_width - 1, y + tile_height - 1], 
                             outline=grid_color[:3], width=3)

            if show_overlap:
                # Draw overlap regions
                if overlap_x > 0 and x > 0:
                    draw.rectangle([x, y, x + overlap_x, y + tile_height], 
                                 fill=overlap_color)
                if overlap_y > 0 and y > 0:
                    draw.rectangle([x, y, x + tile_width, y + overlap_y], 
                                 fill=overlap_color)
                
                # Draw feather zones (larger than overlap)
                feather_x = max(overlap_x, feather_size)
                feather_y = max(overlap_y, feather_size)
                
                if feather_x > overlap_x and x > 0:
                    draw.rectangle([x + overlap_x, y, x + feather_x, y + tile_height], 
                                 fill=feather_color)
                if feather_y > overlap_y and y > 0:
                    draw.rectangle([x, y + overlap_y, x + tile_width, y + feather_y], 
                                 fill=feather_color)

            if show_order:
                # Draw processing order numbers with background
                text_x = x + tile_width // 2
                text_y = y + tile_height // 2
                
                # Draw background circle for better visibility
                draw.ellipse([text_x-20, text_y-20, text_x+20, text_y+20], 
                           fill=(0, 0, 0, 180))
                draw.text((text_x, text_y), str(idx + 1), fill=order_color[:3], anchor="mm")

        # Convert back to tensor
        preview_np = np.array(preview_pil).astype(np.float32) / 255.0
        preview_tensor = torch.tensor(preview_np).unsqueeze(0)

        # Enhanced UI info with debugging details
        blur_radius = calculate_enhanced_blur_radius(overlap_x, overlap_y, egregora_data.get('blur_scale', 0.5))
        
        ui_info = f"""Egregora Tiling Preview (FIXED):
Tiles: {len(tile_coordinates)}
Grid: {grid_x}x{grid_y}
Tile Size: {tile_width}x{tile_height}
Overlap: {overlap_x}x{overlap_y}
Feather: {feather_size}px
Blur Radius: {blur_radius}px (FIXED calculation)
Order: {list(TILE_ORDER_DICT.keys())[tile_order]}
Blending: {egregora_data.get('blending_method', 'advanced_feather')}

Legend:
- Red lines: Tile boundaries
- Green areas: Overlap zones
- Blue areas: Extended feather zones
- Yellow numbers: Processing order"""

        return (preview_tensor, ui_info)

class Egregora_Analyze_Content:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "EGREGORA_ANALYSIS")
    RETURN_NAMES = ("detail_map", "ui", "analysis_data")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Utils"

    def execute(self, image):
        detail_map = ContentAnalyzer.calculate_detail_map(image)
        
        detail_map_norm = (detail_map - detail_map.min()) / (detail_map.max() - detail_map.min() + 1e-8)
        detail_map_rgb = np.stack([detail_map_norm] * 3, axis=2)
        detail_map_tensor = torch.tensor(detail_map_rgb).unsqueeze(0).float()
        
        mean_detail = np.mean(detail_map)
        std_detail = np.std(detail_map)
        max_detail = np.max(detail_map)
        min_detail = np.min(detail_map)
        
        threshold = mean_detail + std_detail
        high_detail_pixels = np.sum(detail_map > threshold)
        total_pixels = detail_map.shape[0] * detail_map.shape[1]
        high_detail_percentage = (high_detail_pixels / total_pixels) * 100
        
        analysis_data = {
            'detail_map': detail_map,
            'mean_detail': mean_detail,
            'std_detail': std_detail,
            'max_detail': max_detail,
            'min_detail': min_detail,
            'high_detail_threshold': threshold,
            'high_detail_percentage': high_detail_percentage
        }
        
        ui_info = f"""Egregora Content Analysis:
Mean Detail: {mean_detail:.4f}
Std Detail: {std_detail:.4f}
Max Detail: {max_detail:.4f}
Min Detail: {min_detail:.4f}
High Detail Regions: {high_detail_percentage:.1f}%

Blending Recommendations:
- Blur Scale: {min(1.0, high_detail_percentage / 50.0):.2f} (higher for complex images)
- Feather Size: {max(32, int(high_detail_percentage * 2))}px
- Method: {'advanced_feather' if high_detail_percentage > 25 else 'distance_field'}
- Overlap: {'Increase to 1/4 tile' if high_detail_percentage > 40 else 'Current is fine'}"""

        return (detail_map_tensor, ui_info, analysis_data)

class Egregora_Turbo_Prompt:
    """Minimal turbo prompt for conditioning"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "caption_text": ("STRING", {"multiline": True, "forceInput": True}),
                "global_positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "global_negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "blacklist_words": ("STRING", {"multiline": True, "default": "", "placeholder": "blacklist_words"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive_conditioning", "negative_conditioning")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Conditioning"

    def _normalize(self, s: str) -> str:
        s = (s or "").strip()
        return re.sub(r"\s+", " ", s)

    def _parse_blacklist(self, text: str):
        if not text:
            return []
        items = [t.strip() for t in re.split(r"[,\n]+", text) if t.strip()]
        seen, out = set(), []
        for w in items:
            lw = w.lower()
            if lw not in seen:
                seen.add(lw)
                out.append(re.escape(w))
        return out

    def _apply_blacklist(self, text: str, terms):
        if not text or not terms:
            return self._normalize(text)
        pattern = r"\b(?:{})\b".format("|".join(terms))
        cleaned = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        return self._normalize(cleaned)

    def _encode(self, clip, text: str):
        tokens = clip.tokenize(text)
        try:
            out = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = out.pop("cond")
            return [[cond, out]]
        except TypeError:
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]

    def execute(self, clip, caption_text, global_positive_prompt, global_negative_prompt, blacklist_words):
        pos_text = self._normalize(f"{caption_text} {global_positive_prompt}".strip())
        terms = self._parse_blacklist(blacklist_words)
        pos_text = self._apply_blacklist(pos_text, terms)

        neg_text = self._normalize(global_negative_prompt)

        positive = self._encode(clip, pos_text if pos_text else " ")
        negative = self._encode(clip, neg_text if neg_text else " ")

        return (positive, negative)

# Node registration
NODE_CLASS_MAPPINGS = {
    "Egregora Algorithm": Egregora_Algorithm,
    "Egregora Divide and Select": Egregora_Divide_Select,
    "Egregora Combine": Egregora_Combine,
    "Egregora Preview": Egregora_Preview,
    "Egregora Analyze Content": Egregora_Analyze_Content,
    "Egregora Turbo Prompt": Egregora_Turbo_Prompt,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Egregora Algorithm": "🧠 Egregora Algorithm",
    "Egregora Divide and Select": "✂️ Egregora Divide & Select",
    "Egregora Combine": "🔗 Egregora Combine",
    "Egregora Preview": "👁️ Egregora Preview",
    "Egregora Analyze Content": "🔍 Egregora Content Analysis",
    "Egregora Turbo Prompt": "🚀 Egregora Turbo Prompt",

}
