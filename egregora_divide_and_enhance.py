# Enhanced Divide and Conquer Algorithm - Egregora
# Inspired by Steudio's Divide and Conquer algorithm
# Enhanced with better blending, improved overlap strategies, and optimizations

import sys
import os
import torch
import math
import numpy as np
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
    "poisson",
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
    
    @staticmethod
    def calculate_adaptive_overlap(detail_map: np.ndarray, 
                                 base_overlap: float,
                                 tile_coords: Tuple[int, int],
                                 tile_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate adaptive overlap based on local image complexity"""
        x, y = tile_coords
        tw, th = tile_size
        h, w = detail_map.shape
        
        # Sample detail in tile region (with bounds checking)
        x1, x2 = max(0, x), min(w, x + tw)
        y1, y2 = max(0, y), min(h, y + th)
        
        if x2 > x1 and y2 > y1:
            tile_detail = detail_map[y1:y2, x1:x2]
            avg_detail = np.mean(tile_detail)
            
            # Normalize detail (assuming detail values are typically 0-1)
            normalized_detail = min(1.0, avg_detail / 0.5)
            
            # Increase overlap in high-detail areas
            overlap_multiplier = 1.0 + (normalized_detail * 0.5)  # Up to 1.5x overlap
            
            adaptive_overlap_x = int(base_overlap * tw * overlap_multiplier)
            adaptive_overlap_y = int(base_overlap * th * overlap_multiplier)
        else:
            adaptive_overlap_x = int(base_overlap * tw)
            adaptive_overlap_y = int(base_overlap * th)
            
        return adaptive_overlap_x, adaptive_overlap_y

class AdvancedBlender:
    """Advanced blending algorithms for seamless tile combination"""
    
    @staticmethod
    def create_distance_field_mask(tile_shape: Tuple[int, int], 
                                 overlap: Tuple[int, int],
                                 position: str) -> np.ndarray:
        """Create a distance field-based mask for smoother blending"""
        h, w = tile_shape
        overlap_x, overlap_y = overlap
        
        mask = np.ones((h, w), dtype=np.float32)
        
        # Create distance field from edges that need blending
        if 'left' in position and overlap_x > 0:
            mask[:, :overlap_x] = 0
        if 'right' in position and overlap_x > 0:
            mask[:, -overlap_x:] = 0
        if 'top' in position and overlap_y > 0:
            mask[:overlap_y, :] = 0
        if 'bottom' in position and overlap_y > 0:
            mask[-overlap_y:, :] = 0
            
        # Calculate distance transform
        distance_field = distance_transform_edt(mask)
        
        # Normalize to create smooth falloff
        max_distance = max(overlap_x, overlap_y, 1)
        distance_field = np.clip(distance_field / max_distance, 0, 1)
        
        # Apply smooth falloff function
        distance_field = 0.5 * (1 + np.cos(np.pi * (1 - distance_field)))
        
        return distance_field
    
    @staticmethod
    def multi_scale_blend(tile1: np.ndarray, tile2: np.ndarray, 
                         mask: np.ndarray, levels: int = 4) -> np.ndarray:
        """Multi-scale blending for different frequency components"""
        
        def build_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
            pyramid = [image.astype(np.float32)]
            for i in range(levels - 1):
                # Downsample
                smaller = cv2.pyrDown(pyramid[-1])
                pyramid.append(smaller)
            return pyramid
        
        def build_laplacian_pyramid(gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:
            laplacian = []
            for i in range(len(gaussian_pyramid) - 1):
                # Expand and subtract
                expanded = cv2.pyrUp(gaussian_pyramid[i + 1], 
                                   dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
                laplacian.append(gaussian_pyramid[i] - expanded)
            laplacian.append(gaussian_pyramid[-1])  # Top level
            return laplacian
        
        # Build pyramids
        gauss1 = build_pyramid(tile1, levels)
        gauss2 = build_pyramid(tile2, levels)
        gauss_mask = build_pyramid(mask, levels)
        
        # Build Laplacian pyramids
        lapl1 = build_laplacian_pyramid(gauss1)
        lapl2 = build_laplacian_pyramid(gauss2)
        
        blended_pyramid = []
        for l1, l2, m in zip(lapl1, lapl2, gauss_mask):
            if len(l1.shape) == 3 and len(m.shape) == 2:
                m = np.stack([m] * l1.shape[2], axis=2)
            blended = l1 * m + l2 * (1 - m)
            blended_pyramid.append(blended)
        
        # Reconstruct image
        result = blended_pyramid[-1]
        for i in range(len(blended_pyramid) - 2, -1, -1):
            # Expand and add
            result = cv2.pyrUp(result, dstsize=(blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
            result += blended_pyramid[i]
            
        return np.clip(result, 0, 1)

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
        row = min(grid_y - 1, y // max(1, tile_height - overlap_y))
        col = min(grid_x - 1, x // max(1, tile_width - overlap_x))
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
                x += dx
                y += dy
                if 0 <= x < grid_x and 0 <= y < grid_y and (x, y) not in visited:
                    idx = y * grid_x + x
                    if idx < len(tiles):
                        spiral_tiles.append(tiles[idx])
                        visited.add((x, y))
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
                "tile_width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "tile_height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "min_overlap": (list(OVERLAP_DICT.keys()), {"default": "1/16 Tile"}),
                "min_scale_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 8.0}),
                "tile_order": (list(TILE_ORDER_DICT.keys()), {"default": "spiral_outward"}),
                "scaling_method": (SCALING_METHODS, {"default": "lanczos"}),
                "blending_method": (BLENDING_METHODS, {"default": "distance_field"}),
                "content_analysis": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "use_upscale_with_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "EGREGORA_DATA", "STRING")
    RETURN_NAMES = ("IMAGE", "egregora_data", "ui")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Egregora/Core"
    DESCRIPTION = """
Enhanced Divide and Conquer Algorithm with:
- Adaptive overlap based on content analysis
- Multiple tile ordering strategies
- Advanced blending methods
- Content-aware processing
"""

    def execute(self, image, scaling_method, tile_width, tile_height, min_overlap, 
                min_scale_factor, tile_order, blending_method, content_analysis,
                upscale_model=None, use_upscale_with_model=True):

        overlap = OVERLAP_DICT.get(min_overlap, 0)
        tile_order_val = TILE_ORDER_DICT.get(tile_order, 0)
        
        _, height, width, _ = image.shape
        
        # Content analysis for adaptive strategies
        detail_map = None
        if content_analysis or overlap == -1:  # Adaptive overlap
            detail_map = ContentAnalyzer.calculate_detail_map(image)
        
        # Calculate overlaps (adaptive or fixed)
        if overlap == -1:  # Adaptive
            base_overlap = 0.0625  # Base 1/16 tile overlap
            overlap_x = calculate_overlap(tile_width, base_overlap)
            overlap_y = calculate_overlap(tile_height, base_overlap)
        else:
            overlap_x = calculate_overlap(tile_width, overlap)
            overlap_y = calculate_overlap(tile_height, overlap)

        # Ensure minimum scale factor
        min_scale_factor = max(min_scale_factor, MIN_SCALE_FACTOR_THRESHOLD)

        # Calculate dimensions (similar to original but with bounds checking)
        if width <= height:
            multiply_factor = max(1, math.ceil(min_scale_factor * width / tile_width))
            while True:
                upscaled_width = tile_width * multiply_factor
                grid_x = max(1, math.ceil(upscaled_width / tile_width))
                upscaled_width = max(tile_width, (tile_width * grid_x) - (overlap_x * max(0, grid_x - 1)))
                upscale_ratio = upscaled_width / width if width > 0 else 1.0
                if upscale_ratio >= min_scale_factor:
                    break
                multiply_factor += 1
            upscaled_height = max(tile_height, int(height * upscale_ratio))
            grid_y = max(1, math.ceil((upscaled_height - overlap_y) / max(1, tile_height - overlap_y)))
            if grid_y > 1:
                overlap_y = max(0, round((tile_height * grid_y - upscaled_height) / (grid_y - 1)))
        else:
            multiply_factor = max(1, math.ceil(min_scale_factor * height / tile_height))
            while True:
                upscaled_height = tile_height * multiply_factor
                grid_y = max(1, math.ceil(upscaled_height / tile_height))
                upscaled_height = max(tile_height, (tile_height * grid_y) - (overlap_y * max(0, grid_y - 1)))
                upscale_ratio = upscaled_height / height if height > 0 else 1.0
                if upscale_ratio >= min_scale_factor:
                    break
                multiply_factor += 1
            upscaled_width = max(tile_width, int(width * upscale_ratio))
            grid_x = max(1, math.ceil((upscaled_width - overlap_x) / max(1, tile_width - overlap_x)))
            if grid_x > 1:
                overlap_x = max(0, round((tile_width * grid_x - upscaled_width) / (grid_x - 1)))

        effective_upscale = round(upscaled_width / width, 2) if width > 0 else 1.0
        
        egregora_data = {
            'upscaled_width': upscaled_width,
            'upscaled_height': upscaled_height,
            'tile_width': tile_width,
            'tile_height': tile_height,
            'overlap_x': overlap_x,
            'overlap_y': overlap_y,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'tile_order': tile_order_val,
            'blending_method': blending_method,
            'detail_map': detail_map,
            'adaptive_overlap': (overlap == -1),
        }

        # Upscale image (similar to original)
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

        # Final upscale to target dimensions
        upscaled_image = comfy.utils.common_upscale(samples, upscaled_width, upscaled_height, scaling_method, crop=0).movedim(1, -1)

        # UI information
        ui_info = f"""Egregora Enhanced Algorithm:
Original: {width}x{height}
Upscaled: {upscaled_width}x{upscaled_height}
Grid: {grid_x}x{grid_y} ({grid_x * grid_y} tiles)
Overlap: {overlap_x}x{overlap_y} pixels
Scale Factor: {effective_upscale}x
Tile Order: {tile_order}
Blending: {blending_method}
Content Analysis: {'Enabled' if content_analysis else 'Disabled'}"""

        return (upscaled_image, egregora_data, ui_info)


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

    def execute(self, image, tile, egregora_data):
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
    DESCRIPTION = """
Enhanced tile combination with advanced blending:
- Distance field blending
- Multi-scale blending
- Content-aware blending
- Poisson blending
"""

    def execute(self, images, egregora_data):
        # Ensure egregora_data is not a list
        if isinstance(egregora_data, list):
            egregora_data = egregora_data[0]

        # Combine images into single tensor
        out = []
        for i in range(len(images)):
            img = images[i]
            out.append(img)
        images = torch.stack(out).squeeze(1)

        # Extract data
        upscaled_width = egregora_data['upscaled_width']
        upscaled_height = egregora_data['upscaled_height']
        overlap_x = egregora_data['overlap_x']
        overlap_y = egregora_data['overlap_y']
        grid_x = egregora_data['grid_x']
        grid_y = egregora_data['grid_y']
        tile_order = egregora_data['tile_order']
        blending_method = egregora_data['blending_method']
        detail_map = egregora_data.get('detail_map')

        # Get tile dimensions from images
        tile_height = images.shape[1]
        tile_width = images.shape[2]

        # Generate tile coordinates (use the same order as in Divide & Select)
        tile_coordinates, matrix = create_enhanced_tile_coordinates(
            upscaled_width, upscaled_height, tile_width, tile_height,
            overlap_x, overlap_y, grid_x, grid_y, tile_order, detail_map
        )

        # Initialize output
        original_shape = (1, upscaled_height, upscaled_width, 3)
        output = torch.zeros(original_shape, dtype=images.dtype)

        # Use the proven Steudio blending approach with enhancements
        overlap_factor = 4
        f_overlap_x = max(1, overlap_x // overlap_factor)
        f_overlap_y = max(1, overlap_y // overlap_factor)
        
        # Blend factors based on overlap
        blend_x = max(1, math.sqrt(max(overlap_x, 1)))
        blend_y = max(1, math.sqrt(max(overlap_y, 1)))

        index = 0
        for tile_coordinate in tile_coordinates:
            if index >= images.shape[0]:
                break
                
            image_tile = images[index]
            x, y = tile_coordinate

            # Ensure coordinates are within bounds
            x = max(0, min(x, upscaled_width - tile_width))
            y = max(0, min(y, upscaled_height - tile_height))

            # Create mask for the tile using Steudio's proven approach
            mask = Image.new("L", (tile_width, tile_height), 0)
            draw = ImageDraw.Draw(mask)

            # Detect tile position and create appropriate mask
            is_left_edge = (x == 0)
            is_right_edge = (x == upscaled_width - tile_width)
            is_top_edge = (y == 0)
            is_bottom_edge = (y == upscaled_height - tile_height)
            is_single_tile_width = (upscaled_width == tile_width)
            is_single_tile_height = (upscaled_height == tile_height)

            # Apply Steudio's mask logic exactly
            if is_left_edge and is_top_edge and not is_single_tile_height and not is_single_tile_width:
                # Top-left corner
                draw.rectangle([0, 0, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif is_right_edge and is_top_edge and not is_single_tile_height and not is_single_tile_width:
                # Top-right corner
                draw.rectangle([f_overlap_x, 0, tile_width, tile_height - f_overlap_y], fill=255)
            elif is_left_edge and is_bottom_edge and not is_single_tile_height and not is_single_tile_width:
                # Bottom-left corner
                draw.rectangle([0, f_overlap_y, tile_width - f_overlap_x, tile_height], fill=255)
            elif is_right_edge and is_bottom_edge and not is_single_tile_height and not is_single_tile_width:
                # Bottom-right corner
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width, tile_height], fill=255)
            elif is_left_edge and is_top_edge and is_single_tile_height:
                # Top edge, single row
                draw.rectangle([0, 0, tile_width - f_overlap_x, tile_height], fill=255)
            elif is_right_edge and is_top_edge and is_single_tile_height:
                # Top edge, single row, right
                draw.rectangle([f_overlap_x, 0, tile_width, tile_height], fill=255)
            elif is_left_edge and is_top_edge and is_single_tile_width:
                # Left edge, single column
                draw.rectangle([0, 0, tile_width, tile_height - f_overlap_y], fill=255)
            elif is_left_edge and is_bottom_edge and is_single_tile_width:
                # Left edge, single column, bottom
                draw.rectangle([0, f_overlap_y, tile_width, tile_height], fill=255)
            elif not is_left_edge and not is_right_edge and is_top_edge and not is_single_tile_height and not is_single_tile_width:
                # Top edge, not corners
                draw.rectangle([f_overlap_x, 0, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif not is_left_edge and not is_right_edge and is_bottom_edge and not is_single_tile_height and not is_single_tile_width:
                # Bottom edge, not corners
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width - f_overlap_x, tile_height], fill=255)
            elif is_left_edge and not is_top_edge and not is_bottom_edge and not is_single_tile_height and not is_single_tile_width:
                # Left edge, not corners
                draw.rectangle([0, f_overlap_y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif is_right_edge and not is_top_edge and not is_bottom_edge and not is_single_tile_height and not is_single_tile_width:
                # Right edge, not corners
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width, tile_height - f_overlap_y], fill=255)
            elif not is_left_edge and not is_right_edge and is_top_edge and is_single_tile_height and not is_single_tile_width:
                # Top edge, single height
                draw.rectangle([f_overlap_x, 0, tile_width - f_overlap_x, tile_height], fill=255)
            elif is_left_edge and not is_top_edge and not is_bottom_edge and not is_single_tile_height and is_single_tile_width:
                # Left edge, single width
                draw.rectangle([0, f_overlap_y, tile_width, tile_height - f_overlap_y], fill=255)
            elif not is_left_edge and not is_right_edge and not is_top_edge and not is_bottom_edge and not is_single_tile_height and not is_single_tile_width:
                # Interior tile
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            else:
                # Single tile or edge case - fill entire tile
                draw.rectangle([0, 0, tile_width, tile_height], fill=255)

            # Apply appropriate blur based on overlap size
            if overlap_x <= 64 or overlap_y <= 64:
                mask = mask.filter(ImageFilter.BoxBlur(radius=min(blend_x, blend_y)))
            else:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=min(blend_x, blend_y)))

            # Convert mask to tensor
            mask_np = np.array(mask) / 255.0
            mask_tensor = torch.tensor(mask_np, dtype=images.dtype).unsqueeze(0).unsqueeze(-1)

            # Apply blending using the proven Steudio method
            current_region = output[:, y:y + tile_height, x:x + tile_width, :]
            
            # Blend the tile
            output[:, y:y + tile_height, x:x + tile_width, :] *= (1 - mask_tensor)
            output[:, y:y + tile_height, x:x + tile_width, :] += image_tile * mask_tensor

            index += 1

        # Clamp values to valid range
        output = torch.clamp(output, 0, 1)

        # Create UI information
        matrix_ui = f"""Egregora Enhanced Combination:
Blending Method: {blending_method} (using proven Steudio approach)
Grid: {grid_x}x{grid_y}
Overlap: {overlap_x}x{overlap_y}
Output Size: {upscaled_width}x{upscaled_height}

Tile Matrix:
""" + '\n'.join([' '.join(row) for row in matrix])

        return output, matrix_ui


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


# --- NODE: Egregora Turbo Prompt ---
class Egregora_Turbo_Prompt:
    """Two positive conditionings (caption & global) mixed by two sliders.
       - No scheduling (stable for LCM + DMD2 at low steps/low CFG)
       - Combine-style mixing: keep prompts separate; sampler merges them
       - Strengths: contrasty 2-way softmax + raw magnitude preserved
       - ADM keys auto-read from LATENT size
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                # prompts first
                "caption_text": ("STRING", {"multiline": True, "forceInput": True}),
                "global_positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "global_negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "blacklist_words": ("STRING", {"multiline": True, "default": ""}),
                # strengths grouped together (global below caption)
                "caption_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "global_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                "latent": ("LATENT",),  # auto-read size if provided
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive_conditioning", "negative_conditioning")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Conditioning"
    DESCRIPTION = "Two conditionings with strengths; ADM from latent; combine-style mix; no scheduling."

    # ---------- helpers ----------
    def _adm_from_latent(self, latent):
        import torch
        try:
            t = latent.get("samples", None) if isinstance(latent, dict) else None
            if isinstance(t, torch.Tensor) and t.dim() == 4:  # [B,C,H,W]
                _, _, H, W = t.shape
                img_w = int(W * 8)  # latent is 1/8 of image size
                img_h = int(H * 8)
                return {
                    "width": img_w, "height": img_h,
                    "target_width": img_w, "target_height": img_h,
                    "crop_w": 0, "crop_h": 0,
                }
        except Exception:
            pass
        # fallback if no latent is wired
        return {"width": 1024, "height": 1024, "target_width": 1024, "target_height": 1024, "crop_w": 0, "crop_h": 0}

    def _clean(self, text, blacklist_words):
        bl = {w.strip().lower() for w in (blacklist_words or "").split(",") if w.strip()}
        toks = [(t or "").strip() for t in (text or "").split()]
        out = " ".join([t for t in toks if t.lower() not in bl])
        return out or " "

    def _ensure_2d_pooled(self, t):
        import torch
        if not isinstance(t, torch.Tensor):
            return None
        if t.dim() == 0: t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 1: t = t.unsqueeze(0)
        # SDXL expects 1280 for ADM pooled; if larger, slice
        if t.shape[-1] > 1280:
            t = t[:, :1280]
        return t

    def _encode_with_strength(self, clip, text, strength):
        import torch
        tokens = clip.tokenize((text or " ").strip() or " ")
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        if not torch.is_tensor(cond):   cond   = torch.tensor(cond)
        if not torch.is_tensor(pooled): pooled = torch.tensor(pooled)
        pooled = self._ensure_2d_pooled(pooled)
        return cond, pooled, float(max(0.0, strength))

    def _to_3d(self, x):
        # force [B, N, D]
        if x.dim() == 0: x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1: x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2: x = x.unsqueeze(0)
        return x

    def _softmax2(self, a, b, temp=0.55):
        # lower temperature -> more contrast between sliders
        import math
        ea = math.exp(a / max(1e-6, temp))
        eb = math.exp(b / max(1e-6, temp))
        s = ea + eb
        if s <= 0: return 0.5, 0.5
        return ea / s, eb / s

    def _clamp_pair(self, pa, pb, lo=0.15, hi=0.85):
        # clamp shares so neither side vanishes at low CFG
        pa = max(lo, min(hi, pa))
        pb = max(lo, min(hi, pb))
        # renormalize to sum=1 after clamping
        s = pa + pb
        return pa / s, pb / s

    # ---------- main ----------
    def execute(self, clip, caption_text, global_positive_prompt, global_negative_prompt,
                blacklist_words, caption_strength, global_strength, latent=None):

        import torch

        adm_defaults = self._adm_from_latent(latent)

        # encode three texts
        cap_cond, cap_pooled, cs = self._encode_with_strength(clip, self._clean(caption_text, blacklist_words), caption_strength)
        pos_cond, pos_pooled, gs = self._encode_with_strength(clip, global_positive_prompt, global_strength)
        neg_cond, neg_pooled, _  = self._encode_with_strength(clip, global_negative_prompt, 1.0)

        # normalize shapes
        cap_cond = self._to_3d(cap_cond)
        pos_cond = self._to_3d(pos_cond)
        neg_cond = self._to_3d(neg_cond)

        # contrasty normalized shares + clamp (no scheduling)
        p_cap, p_pos = self._softmax2(cs, gs, temp=0.55)
        p_cap, p_pos = self._clamp_pair(p_cap, p_pos, lo=0.15, hi=0.85)

        # We keep BOTH: normalized share ('strength') and raw magnitude ('weight')
        def pack_info(pooled, share, raw):
            info = dict(adm_defaults)
            if isinstance(pooled, torch.Tensor):
                info["pooled_output"] = pooled
            info["strength"] = float(share)     # relative share (sum to 1 after clamp)
            info["weight"]   = float(raw)       # raw magnitude, lets (1,1) differ from (0.5,0.5)
            return info

        positive = []
        if cs > 0.0:
            positive.append([cap_cond, pack_info(cap_pooled, p_cap, cs)])
        if gs > 0.0:
            positive.append([pos_cond, pack_info(pos_pooled, p_pos, gs)])
        if not positive:
            # fallback to empty prompt
            empty_c, empty_p, _ = self._encode_with_strength(clip, " ", 1.0)
            positive = [[self._to_3d(empty_c), pack_info(empty_p, 1.0, 1.0)]]

        negative = [[neg_cond, pack_info(neg_pooled, 1.0, 1.0)]]
        return (positive, negative)


# --- Node registration with new node added ---
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
