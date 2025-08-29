# Enhanced Divide and Conquer Algorithm - Egregora (FIXED)
# Inspired by Steudio's Divide and Conquer algorithm
# Enhanced with better blending, improved overlap strategies, and optimizations

import sys
import os
import torch
import math
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import comfy.utils
from comfy import model_management
import cv2
from scipy.ndimage import distance_transform_edt
from typing import Tuple, List, Dict, Optional

OVERLAP_DICT = {
    "None": 0,
    "1/64 Tile": 0.015625,
    "1/32 Tile": 0.03125,
    "1/16 Tile": 0.0625,
    "1/8 Tile": 0.125,
    "1/4 Tile": 0.25,
    "1/2 Tile": 0.5,
    "Adaptive": -1,  # Special value for adaptive overlap
}

TILE_ORDER_DICT = {
    "linear": 0,
    "spiral_outward": 1,
    "spiral_inward": 2,
    "serpentine": 3,
    "content_aware": 4,
    "dependency_optimized": 5
}

BLENDING_METHODS = [
    "gaussian_blur",
    "multi_scale",
    "distance_field",
    "frequency_domain"
]

SCALING_METHODS = [
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos"
]

MIN_SCALE_FACTOR_THRESHOLD = 1.0

class ContentAnalyzer:
    """Analyzes image content to optimize tiling strategy"""
    
    @staticmethod
    def calculate_detail_map(image_tensor: torch.Tensor) -> np.ndarray:
        """Calculate detail/complexity map of image using gradient magnitude"""
        # Convert to numpy and to grayscale
        image_np = image_tensor.squeeze(0).cpu().numpy()
        if len(image_np.shape) == 3:
            gray = np.dot(image_np, [0.299, 0.587, 0.114])
        else:
            gray = image_np
            
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply Gaussian blur to smooth the detail map
        detail_map = cv2.GaussianBlur(gradient_magnitude, (15, 15), 0)
        
        return detail_map

def calculate_overlap(tile_size, overlap_fraction):
    """Calculate overlap with bounds checking"""
    return max(0, int(overlap_fraction * tile_size))

def create_enhanced_tile_coordinates(image_width: int, image_height: int, 
                                   tile_width: int, tile_height: int,
                                   overlap_x: int, overlap_y: int,
                                   grid_x: int, grid_y: int,
                                   tile_order: int,
                                   detail_map: Optional[np.ndarray] = None) -> Tuple[List[Tuple[int, int]], List[List[str]]]:
    """Enhanced tile coordinate generation with multiple ordering strategies"""
    
    tiles = []
    matrix = [['' for _ in range(grid_x)] for _ in range(grid_y)]
    
    # Generate basic tile coordinates
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
        # Find grid position (approximate due to potential coordinate adjustments)
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
    
    # Start from center
    cx, cy = grid_x // 2, grid_y // 2
    x, y = cx, cy
    dx, dy = 1, 0
    layer = 1
    
    # Add center tile
    if 0 <= x < grid_x and 0 <= y < grid_y:
        idx = y * grid_x + x
        if idx < len(tiles):
            spiral_tiles.append(tiles[idx])
            visited.add((x, y))
    
    while len(spiral_tiles) < len(tiles):
        for _ in range(2):  # Two sides per layer
            for _ in range(layer):
                if 0 <= x < grid_x and 0 <= y < grid_y and (x, y) not in visited:
                    idx = y * grid_x + x
                    if idx < len(tiles):
                        spiral_tiles.append(tiles[idx])
                        visited.add((x, y))
                x += dx
                y += dy
            dx, dy = -dy, dx  # Rotate clockwise
        layer += 1
    
    return spiral_tiles if outward else spiral_tiles[::-1]

def _serpentine_order(tiles: List[Tuple[int, int]], grid_x: int, grid_y: int) -> List[Tuple[int, int]]:
    """Generate serpentine (snake-like) tile order"""
    serpentine_tiles = []
    
    for row in range(grid_y):
        if row % 2 == 0:  # Left to right
            for col in range(grid_x):
                idx = row * grid_x + col
                if idx < len(tiles):
                    serpentine_tiles.append(tiles[idx])
        else:  # Right to left
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
        
        # Sample detail in tile region
        x1, x2 = max(0, x), min(w, x + tile_width)
        y1, y2 = max(0, y), min(h, y + tile_height)
        
        if x2 > x1 and y2 > y1:
            return np.mean(detail_map[y1:y2, x1:x2])
        return 0.0
    
    # Sort by complexity (descending)
    tiles_with_complexity = [(tile, get_tile_complexity(tile)) for tile in tiles]
    tiles_with_complexity.sort(key=lambda x: x[1], reverse=True)
    
    return [tile for tile, _ in tiles_with_complexity]

def _dependency_optimized_order(tiles: List[Tuple[int, int]], grid_x: int, grid_y: int) -> List[Tuple[int, int]]:
    """Order tiles to minimize dependencies (corner tiles first, then edges, then interior)"""
    
    def get_tile_priority(idx):
        row, col = idx // grid_x, idx % grid_x
        
        # Corners have highest priority
        if (row == 0 or row == grid_y - 1) and (col == 0 or col == grid_x - 1):
            return 0
        # Edges have medium priority
        elif row == 0 or row == grid_y - 1 or col == 0 or col == grid_x - 1:
            return 1
        # Interior tiles have lowest priority
        else:
            return 2
    
    # Sort by priority
    indexed_tiles = [(i, tiles[i], get_tile_priority(i)) for i in range(len(tiles))]
    indexed_tiles.sort(key=lambda x: (x[2], x[0]))  # Sort by priority, then by original index
    
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
                "blending_method": (["gaussian_blur", "distance_field", "feathered"], {"default": "distance_field"}),
                "blend_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blur_scale": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "feather_falloff": (["linear", "exponential", "smooth"], {"default": "smooth"}),
                "content_analysis": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "use_upscale_with_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "EGREGORA_DATA", "STRING")
    RETURN_NAMES = ("IMAGE", "egregora_data", "ui")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"
    DESCRIPTION = """
Enhanced Divide and Conquer Algorithm with:
- Robust overlap calculation with bounds checking
- Optimized tile ordering strategies  
- Reliable blending methods (gaussian_blur, distance_field, feathered)
- Optional content-aware processing
- Improved memory management
"""

    def calculate_grid_dimensions(self, width, height, tile_width, tile_height, 
                                 overlap_x, overlap_y, min_scale_factor):
        """Fixed aspect ratio preserving grid calculation based on DaC approach"""
        
        # Ensure minimum scale factor
        min_scale_factor = max(min_scale_factor, MIN_SCALE_FACTOR_THRESHOLD)
        
        # Use the same logic as the original DaC implementation
        if width <= height:
            # Calculate based on width first (portrait or square)
            multiply_factor = math.ceil(min_scale_factor * width / tile_width)
            while True:
                upscaled_width = tile_width * multiply_factor
                grid_x = math.ceil(upscaled_width / tile_width)
                upscaled_width = (tile_width * grid_x) - (overlap_x * (grid_x - 1))
                upscale_ratio = upscaled_width / width
                if upscale_ratio >= min_scale_factor:
                    break
                multiply_factor += 1
            
            # Calculate height maintaining aspect ratio
            upscaled_height = int(height * upscale_ratio)
            
            # Calculate grid_y based on the calculated height
            grid_y = math.ceil((upscaled_height - overlap_y) / (tile_height - overlap_y))
            
            # Recalculate overlap_y to fit exactly
            if grid_y > 1:
                overlap_y = round((tile_height * grid_y - upscaled_height) / (grid_y - 1))
            else:
                overlap_y = 0
                
        else:
            # Calculate based on height first (landscape)
            multiply_factor = math.ceil(min_scale_factor * height / tile_height)
            while True:
                upscaled_height = tile_height * multiply_factor
                grid_y = math.ceil(upscaled_height / tile_height)
                upscaled_height = (tile_height * grid_y) - (overlap_y * (grid_y - 1))
                upscale_ratio = upscaled_height / height
                if upscale_ratio >= min_scale_factor:
                    break
                multiply_factor += 1
            
            # Calculate width maintaining aspect ratio
            upscaled_width = int(width * upscale_ratio)
            
            # Calculate grid_x based on the calculated width
            grid_x = math.ceil((upscaled_width - overlap_x) / (tile_width - overlap_x))
            
            # Recalculate overlap_x to fit exactly
            if grid_x > 1:
                overlap_x = round((tile_width * grid_x - upscaled_width) / (grid_x - 1))
            else:
                overlap_x = 0
        
        # Ensure all values are positive and reasonable
        upscaled_width = max(tile_width, int(upscaled_width))
        upscaled_height = max(tile_height, int(upscaled_height))
        grid_x = max(1, grid_x)
        grid_y = max(1, grid_y)
        overlap_x = max(0, min(overlap_x, tile_width - 1))
        overlap_y = max(0, min(overlap_y, tile_height - 1))
        
        return upscaled_width, upscaled_height, grid_x, grid_y, overlap_x, overlap_y

    def validate_parameters(self, tile_width, tile_height, overlap_x, overlap_y, grid_x, grid_y):
        """Validate all parameters are within reasonable bounds"""
        
        # Check tile dimensions
        if tile_width < 64 or tile_height < 64:
            raise ValueError("Tile dimensions must be at least 64x64")
        
        if tile_width > 4096 or tile_height > 4096:
            raise ValueError("Tile dimensions must be at most 4096x4096")
        
        # Check overlap doesn't exceed tile size
        if overlap_x >= tile_width:
            overlap_x = max(0, tile_width - 32) 
            
        if overlap_y >= tile_height:
            overlap_y = max(0, tile_height - 32)
            
        # Check grid size is reasonable
        if grid_x * grid_y > 100: 
            raise ValueError(f"Grid size {grid_x}x{grid_y} = {grid_x * grid_y} tiles is too large (max 100)")
            
        return overlap_x, overlap_y

    def execute(self, image, tile_width, tile_height, min_overlap, min_scale_factor, 
                tile_order, scaling_method, blending_method, blend_strength, blur_scale, feather_falloff, content_analysis,
                upscale_model=None, use_upscale_with_model=True):

        try:
            # Get dimensions
            _, height, width, _ = image.shape
            
            # Parse overlap
            overlap = OVERLAP_DICT.get(min_overlap, 0.125)
            tile_order_val = TILE_ORDER_DICT.get(tile_order, 0)
            
            # Calculate initial overlap values
            if overlap == -1: 
                base_overlap = 0.0625
                overlap_x = calculate_overlap(tile_width, base_overlap)
                overlap_y = calculate_overlap(tile_height, base_overlap)
            else:
                overlap_x = calculate_overlap(tile_width, overlap)
                overlap_y = calculate_overlap(tile_height, overlap)
            
            # Calculate robust grid dimensions
            upscaled_width, upscaled_height, grid_x, grid_y, overlap_x, overlap_y = \
                self.calculate_grid_dimensions(width, height, tile_width, tile_height,
                                                overlap_x, overlap_y, min_scale_factor)
            
            # Validate parameters
            overlap_x, overlap_y = self.validate_parameters(
                tile_width, tile_height, overlap_x, overlap_y, grid_x, grid_y
            )
            
            # Content analysis (only if requested and useful)
            detail_map = None
            if content_analysis and (overlap == -1 or tile_order_val == 4): 
                try:
                    detail_map = ContentAnalyzer.calculate_detail_map(image)
                    # Resize detail map to match upscaled dimensions immediately
                    if detail_map.shape != (upscaled_height, upscaled_width):
                        detail_map = cv2.resize(
                            detail_map.astype(np.float32),
                            (upscaled_width, upscaled_height),
                            interpolation=cv2.INTER_LINEAR
                        )
                except Exception as e:
                    print(f"Content analysis failed, proceeding without: {e}")
                    detail_map = None
            
            # Calculate effective upscale ratio
            effective_upscale = round(max(upscaled_width / width, upscaled_height / height), 2) if width > 0 and height > 0 else 1.0
            
            # Create egregora_data with validated values
            egregora_data = {
                'upscaled_width': int(upscaled_width),
                'upscaled_height': int(upscaled_height),
                'tile_width': int(tile_width),
                'tile_height': int(tile_height),
                'overlap_x': int(overlap_x),
                'overlap_y': int(overlap_y),
                'grid_x': int(grid_x),
                'grid_y': int(grid_y),
                'tile_order': int(tile_order_val),
                'blending_method': str(blending_method),
                'blend_strength': float(blend_strength),
                'blur_scale': float(blur_scale),
                'feather_falloff': str(feather_falloff),
                'detail_map': detail_map,
                'adaptive_overlap': (overlap == -1),
            }

            # Upscale image using existing proven method
            if use_upscale_with_model and upscale_model:
                device = model_management.get_torch_device()
                memory_required = model_management.module_size(upscale_model.model)
                memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
                memory_required += image.nelement() * image.element_size()
                model_management.free_memory(memory_required, device)

                upscale_model.to(device)
                in_img = image.movedim(-1, -3).to(device)

                tile = 512
                overlap_value = 32

                oom = True
                while oom:
                    try:
                        steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap_value)
                        pbar = comfy.utils.ProgressBar(steps)
                        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap_value, upscale_amount=upscale_model.scale, pbar=pbar)
                        oom = False
                    except model_management.OOM_EXCEPTION as e:
                        tile //= 2
                        if tile < 128:
                            raise e

                upscale_model.to("cpu")
                upscaled_with_model = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
                samples = upscaled_with_model.movedim(-1, 1)
            else:
                samples = image.movedim(-1, 1)

            # Final upscale to exact target dimensions
            upscaled_image = comfy.utils.common_upscale(samples, upscaled_width, upscaled_height, scaling_method, crop=0).movedim(1, -1)

            # Create comprehensive UI information
            ui_info = f"""Egregora Enhanced Algorithm:
Original: {width}x{height}
Upscaled: {upscaled_width}x{upscaled_height} ({effective_upscale}x)
Grid: {grid_x}x{grid_y} ({grid_x * grid_y} tiles)
Tile Size: {tile_width}x{tile_height}
Overlap: {overlap_x}x{overlap_y} pixels ({overlap_x/tile_width:.1%} x {overlap_y/tile_height:.1%})
Tile Order: {tile_order}
Blending: {blending_method}
Blend Strength: {blend_strength:.2f}
Blur Scale: {blur_scale:.2f}
Feather Falloff: {feather_falloff}
Content Analysis: {'Enabled' if content_analysis and detail_map is not None else 'Disabled'}
Adaptive Overlap: {'Yes' if overlap == -1 else 'No'}

Memory Efficiency: {int(tile_width * tile_height / 1024)}K pixels per tile
Processing Load: {grid_x * grid_y} iterations"""

            return (upscaled_image, egregora_data, ui_info)
            
        except Exception as e:
            # Provide helpful error information
            error_msg = f"""Egregora Algorithm Error: {str(e)}

Parameters when error occurred:
- Image: {width}x{height} 
- Tile: {tile_width}x{tile_height}
- Min Scale: {min_scale_factor}x
- Overlap: {min_overlap}

Try reducing tile size or scale factor."""
            
            # Return a safe fallback
            fallback_image = torch.zeros((1, max(512, height), max(512, width), 3))
            fallback_data = {
                'upscaled_width': max(512, width),
                'upscaled_height': max(512, height), 
                'tile_width': 512,
                'tile_height': 512,
                'overlap_x': 64,
                'overlap_y': 64,
                'grid_x': 1,
                'grid_y': 1,
                'tile_order': 0,
                'blending_method': 'distance_field',
                'detail_map': None,
                'adaptive_overlap': False,
            }
            return (fallback_image, fallback_data, error_msg)

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
    DESCRIPTION = """
Enhanced tile selection with content-aware ordering:
tile 0 = All tiles
tile # = Specific tile #
"""

    def execute(self, image, egregora_data, tile):
        image_height = image.shape[1]
        image_width = image.shape[2]

        # Extract data
        tile_width = egregora_data['tile_width']
        tile_height = egregora_data['tile_height']
        overlap_x = egregora_data['overlap_x']
        overlap_y = egregora_data['overlap_y']
        grid_x = egregora_data['grid_x']
        grid_y = egregora_data['grid_y']
        tile_order = egregora_data['tile_order']
        detail_map = egregora_data.get('detail_map')

        # Generate enhanced tile coordinates
        tile_coordinates, matrix = create_enhanced_tile_coordinates(
            image_width, image_height, tile_width, tile_height, 
            overlap_x, overlap_y, grid_x, grid_y, tile_order, detail_map
        )

        # Extract tiles
        image_tiles = []
        for tile_coord in tile_coordinates:
            x, y = tile_coord
            
            # Ensure coordinates are within bounds
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
            # Return empty result
            return ([torch.zeros((1, tile_height, tile_width, 3))], "No tiles generated")

        all_tiles = torch.cat(image_tiles, dim=0)

        if tile == 0:
            tile_or_tiles = all_tiles
        else:
            if tile <= len(image_tiles):
                tile_or_tiles = image_tiles[tile - 1]
            else:
                tile_or_tiles = image_tiles[0]  # Fallback to first tile

        # Create matrix UI
        matrix_ui = "Egregora Tile Matrix:\n" + '\n'.join([' '.join(row) for row in matrix])

        return ([tile_or_tiles[i].unsqueeze(0) for i in range(tile_or_tiles.shape[0])], matrix_ui)

class Egregora_Combine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "egregora_data": ("EGREGORA_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "ui")
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"
    DESCRIPTION = "Seam-safe normalized blending with working blur_scale, blend_strength and falloff."

    # ---------- helpers ----------
    def _linear_ramp_mask(self, th, tw, ox, oy, has_left, has_right, has_top, has_bottom):
        import numpy as np
        w = np.ones((th, tw), dtype=np.float32)
        if ox > 0 and has_left:
            grad = np.linspace(0.0, 1.0, ox, dtype=np.float32)
            w[:, :ox] *= grad[None, :]
        if ox > 0 and has_right:
            grad = np.linspace(1.0, 0.0, ox, dtype=np.float32)
            w[:, -ox:] *= grad[None, :]
        if oy > 0 and has_top:
            grad = np.linspace(0.0, 1.0, oy, dtype=np.float32)
            w[:oy, :] *= grad[:, None]
        if oy > 0 and has_bottom:
            grad = np.linspace(1.0, 0.0, oy, dtype=np.float32)
            w[-oy:, :] *= grad[:, None]
        return w

    def _distance_field_mask(self, base_mask, blur_px):
        import numpy as np
        from scipy.ndimage import distance_transform_edt
        # EDT on the binary "inside" -> normalize to 0..1 toward edges
        inside = (base_mask > 0.0).astype(np.uint8)
        dt = distance_transform_edt(inside).astype(np.float32)
        mx = float(dt.max())
        m = (dt / mx) if mx > 0 else base_mask.copy()
        if blur_px > 0:
            import cv2
            k = blur_px * 2 + 1
            m = cv2.GaussianBlur(m, (k, k), 0)
        return np.clip(m, 1e-6, 1.0)

    def _apply_falloff_and_strength(self, m, blend_strength, falloff):
        import numpy as np
        # Strength: 0 = hard, 1 = soft
        bs = float(max(0.0, min(1.0, blend_strength)))
        gamma = 6.0 - 5.0 * bs                # harder -> larger gamma
        m = np.power(np.clip(m, 1e-6, 1.0), gamma)
        if falloff == "exponential":
            m = np.power(m, 1.5)
        elif falloff == "smooth":
            m = m * m * (3.0 - 2.0 * m)
        return np.clip(m, 1e-6, 1.0)

    def _final_blur(self, m, blur_px):
        # Global smoothing so blur_scale affects ALL methods
        if blur_px <= 0:
            return m
        import cv2, numpy as np
        k = blur_px * 2 + 1
        return cv2.GaussianBlur(m.astype(np.float32), (k, k), 0)

    # ---------- main ----------
    def execute(self, images, egregora_data):
        import torch, numpy as np

        if isinstance(egregora_data, (list, tuple)):
            egregora_data = egregora_data[0]

        # flatten list of tensors -> (N,H,W,C)
        tiles_list = []
        for itm in images:
            if isinstance(itm, (list, tuple)):
                tiles_list.extend(itm)
            else:
                tiles_list.append(itm)
        tiles = torch.cat([t if t.dim() == 4 else t.unsqueeze(0) for t in tiles_list], dim=0)
        device, dtype = tiles.device, tiles.dtype

        # params
        up_w = int(egregora_data["upscaled_width"])
        up_h = int(egregora_data["upscaled_height"])
        tw   = int(egregora_data["tile_width"])
        th   = int(egregora_data["tile_height"])
        ox   = int(egregora_data["overlap_x"])
        oy   = int(egregora_data["overlap_y"])
        gx   = int(egregora_data["grid_x"])
        gy   = int(egregora_data["grid_y"])
        order = int(egregora_data.get("tile_order", 0))
        detail_map = egregora_data.get("detail_map")

        blending_method = str(egregora_data.get("blending_method", "distance_field"))
        # Explicitly ignore any 'poisson' value (UI removed)
        if blending_method == "poisson":
            blending_method = "distance_field"

        blend_strength  = float(egregora_data.get("blend_strength", 0.4))
        blur_scale      = float(egregora_data.get("blur_scale", 0.3))
        falloff         = str(egregora_data.get("feather_falloff", "linear"))

        # tile coords respecting order used in Divide
        coords, _ = create_enhanced_tile_coordinates(
            up_w, up_h, tw, th, ox, oy, gx, gy, order, detail_map
        )

        # accumulators for normalized blending
        acc = torch.zeros((up_h, up_w, 3), dtype=dtype, device=device)
        ws  = torch.zeros((up_h, up_w, 1), dtype=dtype, device=device)

        max_ov = max(ox, oy)
        # make blur radius from blur_scale; require >0 and odd kernel
        blur_px = int(round(max_ov * max(0.0, min(1.0, blur_scale))))

        for i, (x, y) in enumerate(coords):
            if i >= tiles.shape[0]:
                break
            t = tiles[i].squeeze(0).clamp(0, 1)

            has_left   = (x > 0)
            has_right  = (x + tw < up_w)
            has_top    = (y > 0)
            has_bottom = (y + th < up_h)

            base = self._linear_ramp_mask(th, tw, ox, oy, has_left, has_right, has_top, has_bottom)

            # ---- per-method mask ----
            if blending_method == "gaussian_blur":
                # BUG FIX: blur the *base mask*, not a constant ones array
                import cv2
                m = base
                if blur_px > 0:
                    k = blur_px * 2 + 1
                    m = cv2.GaussianBlur(m.astype(np.float32), (k, k), 0)
            elif blending_method == "distance_field":
                m = self._distance_field_mask(base, blur_px=0)  # DF itself first
            elif blending_method == "feathered":
                m = base  # simple feather from overlaps
            else:
                m = base  # fallback safe

            # global blur so blur_scale affects ALL modes
            m = self._final_blur(m, blur_px)

            # falloff + strength shaping
            m = self._apply_falloff_and_strength(m, blend_strength, falloff)

            # torch compose (normalized, seam-safe)
            w = torch.from_numpy(m).to(device=device, dtype=dtype).unsqueeze(-1)
            acc[y:y+th, x:x+tw, :] += t * w
            ws [y:y+th, x:x+tw, :] += w

        out = acc / torch.clamp(ws, min=1e-6)
        out = out.unsqueeze(0)

        # UI
        try:
            order_name = list(TILE_ORDER_DICT.keys())[order]
        except Exception:
            order_name = str(order)
        ui = (
            f"Egregora Combine\n"
            f"Upscaled: {up_w}x{up_h}\n"
            f"Grid: {gx}x{gy}  Tiles: {gx*gy}\n"
            f"Tile: {tw}x{th}  Overlap: {ox}x{oy}\n"
            f"Order: {order_name}\n"
            f"Blending: {blending_method}\n"
            f"Blend Strength: {blend_strength:.2f}\n"
            f"Blur Scale: {blur_scale:.2f}\n"
            f"Falloff: {falloff}"
        )
        return (out, ui)

            
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
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "ui")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Utils"
    DESCRIPTION = """
Preview the tiling strategy before processing:
- Shows tile boundaries
- Visualizes overlap regions
- Displays processing order
"""

    def execute(self, image, egregora_data, show_grid, show_overlap, show_order):
        # Get image dimensions
        _, height, width, _ = image.shape
        
        # Extract data
        tile_width = egregora_data['tile_width']
        tile_height = egregora_data['tile_height']
        overlap_x = egregora_data['overlap_x']
        overlap_y = egregora_data['overlap_y']
        grid_x = egregora_data['grid_x']
        grid_y = egregora_data['grid_y']
        tile_order = egregora_data['tile_order']
        detail_map = egregora_data.get('detail_map')

        # Generate tile coordinates
        tile_coordinates, matrix = create_enhanced_tile_coordinates(
            width, height, tile_width, tile_height,
            overlap_x, overlap_y, grid_x, grid_y, tile_order, detail_map
        )

        # Create preview image
        preview = image.clone()
        preview_np = (preview.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        preview_pil = Image.fromarray(preview_np)
        draw = ImageDraw.Draw(preview_pil, 'RGBA')

        # Colors for visualization
        grid_color = (255, 0, 0, 128)  # Semi-transparent red
        overlap_color = (0, 255, 0, 64)  # Semi-transparent green
        order_color = (255, 255, 0, 180)  # Semi-transparent yellow

        for idx, (x, y) in enumerate(tile_coordinates):
            if show_grid:
                # Draw tile boundary
                draw.rectangle([x, y, x + tile_width - 1, y + tile_height - 1], 
                             outline=grid_color[:3], width=2)

            if show_overlap:
                # Draw overlap regions
                if overlap_x > 0 and x > 0:
                    draw.rectangle([x, y, x + overlap_x, y + tile_height], 
                                 fill=overlap_color)
                if overlap_y > 0 and y > 0:
                    draw.rectangle([x, y, x + tile_width, y + overlap_y], 
                                 fill=overlap_color)

            if show_order:
                # Draw processing order numbers
                text_x = x + tile_width // 2
                text_y = y + tile_height // 2
                draw.text((text_x, text_y), str(idx + 1), fill=order_color[:3], anchor="mm")

        # Convert back to tensor
        preview_np = np.array(preview_pil).astype(np.float32) / 255.0
        preview_tensor = torch.tensor(preview_np).unsqueeze(0)

        # Create UI info
        ui_info = f"""Egregora Tiling Preview:
Tiles: {len(tile_coordinates)}
Grid: {grid_x}x{grid_y}
Tile Size: {tile_width}x{tile_height}
Overlap: {overlap_x}x{overlap_y}
Order: {list(TILE_ORDER_DICT.keys())[tile_order]}"""

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
    DESCRIPTION = """
Analyze image content for optimal tiling strategy:
- Generate detail/complexity map
- Identify high-detail regions
- Suggest optimal tile placement
"""

    def execute(self, image):
        # Calculate detail map
        detail_map = ContentAnalyzer.calculate_detail_map(image)
        
        # Normalize for visualization
        detail_map_norm = (detail_map - detail_map.min()) / (detail_map.max() - detail_map.min() + 1e-8)
        
        # Convert to RGB for visualization
        detail_map_rgb = np.stack([detail_map_norm] * 3, axis=2)
        detail_map_tensor = torch.tensor(detail_map_rgb).unsqueeze(0).float()
        
        # Calculate statistics
        mean_detail = np.mean(detail_map)
        std_detail = np.std(detail_map)
        max_detail = np.max(detail_map)
        min_detail = np.min(detail_map)
        
        # Find high-detail regions (above mean + std)
        threshold = mean_detail + std_detail
        high_detail_pixels = np.sum(detail_map > threshold)
        total_pixels = detail_map.shape[0] * detail_map.shape[1]
        high_detail_percentage = (high_detail_pixels / total_pixels) * 100
        
        # Analysis data for other nodes
        analysis_data = {
            'detail_map': detail_map,
            'mean_detail': mean_detail,
            'std_detail': std_detail,
            'max_detail': max_detail,
            'min_detail': min_detail,
            'high_detail_threshold': threshold,
            'high_detail_percentage': high_detail_percentage
        }
        
        # Create UI info
        ui_info = f"""Egregora Content Analysis:
Mean Detail: {mean_detail:.4f}
Std Detail: {std_detail:.4f}
Max Detail: {max_detail:.4f}
Min Detail: {min_detail:.4f}
High Detail Regions: {high_detail_percentage:.1f}%

Recommendations:
- {'Increase overlap in high-detail areas' if high_detail_percentage > 25 else 'Standard overlap should suffice'}
- {'Use content-aware tile ordering' if high_detail_percentage > 15 else 'Standard ordering is fine'}
- {'Consider smaller tiles' if max_detail > mean_detail * 3 else 'Current tile size appropriate'}"""

        return (detail_map_tensor, ui_info, analysis_data)


class Egregora_Turbo_Prompt:
    """
    Minimal turbo prompt:
      - POSITIVE = caption_text + global_positive_prompt (plain concat)
      - NEGATIVE = global_negative_prompt
      - blacklist_words (text): comma/newline-separated terms removed from POSITIVE
      - No strengths, no extra toggles, no separators
      - Outputs CONDITIONING with pooled_output
    """

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
    DESCRIPTION = "Concatenate caption+global positive; use global negative; optional blacklist removal."

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
        # Prefer newer API that returns a dict; fall back if unavailable
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


# --- Node registration ---
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
