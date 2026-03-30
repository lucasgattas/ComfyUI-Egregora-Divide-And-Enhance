# ComfyUI · Egregora: Divide & Enhance 🧩✨

A focused set of ComfyUI nodes for tiled image processing, designed to split images into owner-based padded tiles, generate seam-aware masks, and recombine them with smoother, more natural transitions.

**Egregora: Divide & Enhance** is built for high-quality tiled upscaling and enhancement workflows where a large image must be processed in parts without losing spatial consistency.

## 🌟 Core features

- 🧠 deterministic tile planning  
- 🧩 owner-based base regions  
- 🖼️ padded tiles for context  
- 🎭 mask-driven recombination  
- 🌊 smoother seam transitions with optional mask warping  

---

## 📌 What this node pack is for

These nodes are designed for workflows where processing the full image in one pass is too expensive, too memory-heavy, or too unstable.

Typical use cases:

- tiled upscaling
- large-image enhancement
- img2img or detail passes on very large images
- VRAM-constrained workflows
- mask-aware recombination of processed tiles
- debugging tile masks before expensive runs

The main goal is not just to split an image, but to split it in a way that makes the merge more stable and more natural afterward.

---

## 🏗️ Core idea

The system no longer treats tiles as large overlapping windows competing equally over the same semantic region.

Instead, it uses:

- **base ownership regions**  
  each tile has a primary region it owns

- **context padding**  
  the tile is expanded beyond its base region so the model has neighboring visual context

- **mask-based seam blending**  
  only the padded border area is used for the transition

This makes tiled processing more stable, especially in stronger enhancement workflows where neighboring tiles may otherwise drift apart too much.

---

## 🧱 Included nodes

### 🚀 Egregora Algorithm

Builds the tile plan and the working upscaled canvas.

**Inputs**
- `image`
- `tile_resolution`
- `padding_px`
- `min_scale_factor`
- `tile_order`
- `scaling_method`

**Outputs**
- `IMAGE`
- `EGREGORA_DATA`

**What it does**
- computes the working resolution from the input image and `min_scale_factor`
- rescales the image to that working size
- divides the image into a deterministic base grid
- expands each base tile with context padding
- stores all tile placement data in `EGREGORA_DATA`

**Why it matters**
- split and merge always use the same tile plan
- ownership and padding stay consistent
- tile placement remains deterministic and reproducible

---

### ✂️ Egregora Divide Select

Splits the image into padded tiles and generates the masks used later in recombination.

**Inputs**
- `image`
- `egregora_data`
- `tile`
- `blend_px`
- `feather_curve`
- `mask_warp_strength`
- `mask_warp_frequency`

**Outputs**
- `TILE(S)`
- `MASK(S)`

**What it does**
- outputs all tiles or one selected tile
- generates masks that correspond exactly to the padded tile layout
- uses a blend band around the owner region for transition
- can slightly warp the mask to avoid perfectly straight seam lines

**Important behavior**
- `tile = 0` returns all tiles and all masks
- any positive tile index returns only the selected tile and its mask

---

### 🔗 Egregora Combine

Recombines processed tiles using the masks generated during the split stage.

**Inputs**
- `tiles`
- `masks`
- `egregora_data`
- `scaling_method`

**Outputs**
- `IMAGE`

**What it does**
- places each processed tile back into its exact padded location
- applies the provided masks directly
- accumulates image values and mask weights
- normalizes the final image so the result is blended cleanly

**Why this matters**
The combine stage does not rebuild masks from scratch.  
It uses the masks already generated during the divide stage, which keeps split and merge behavior aligned.

---

### 🧪 Egregora Debug Mask

Outputs the generated masks directly for inspection.

**Inputs**
- `egregora_data`
- `blend_px`
- `feather_curve`
- `mask_warp_strength`
- `mask_warp_frequency`
- `tile_index`

**Outputs**
- `MASK`

**What it does**
- previews the masks exactly as they are being generated
- lets you inspect all masks or a single mask
- helps debug seam behavior before running expensive processing

**Important behavior**
- `tile_index = 0` returns the full batch of masks
- any positive tile index returns one selected mask

---

## 🔄 Current node flow

### Standard workflow

```text
Egregora Algorithm
    ↓
Egregora Divide Select
    ↓
[process tiles with your own workflow]
    ↓
Egregora Combine
```

### Mask inspection workflow

```text
Egregora Algorithm
    ↓
Egregora Debug Mask
```

---

## 🧠 Main concepts

### 1. Base ownership
Each tile has a base region it owns.  
This avoids excessive semantic competition between neighboring tiles.

### 2. Padding for context
Each tile is expanded beyond its base region.  
This gives the model more context while processing the tile.

### 3. Blend band
The transition between tiles happens in a controllable band defined by `blend_px`.

### 4. Mask warping
The mask can be gently warped to reduce perfectly straight seam lines.

---

## ⚙️ Important parameters

### `tile_resolution`
Controls the base size of each tile region.

Larger values:
- reduce the total number of tiles
- need more VRAM
- often improve consistency

Smaller values:
- use less VRAM
- create more seams
- can increase tile-to-tile variation

### `padding_px`
Controls how much contextual padding is added around each base tile.

Higher values:
- give the model more neighboring context
- usually improve continuity
- increase compute and overlap cost

### `blend_px`
Controls the width of the transition band around the owner region.

Higher values:
- create longer, softer transitions
- can reduce visible seams
- may increase blending between more different tiles

Lower values:
- keep ownership more strict
- may make transitions more visible

### `feather_curve`
Controls the shape of the blend falloff.

Available options:
- `linear`
- `smoothstep`
- `smootherstep`
- `cosine`

General guidance:
- `linear` is the most direct
- `smoothstep` is softer
- `smootherstep` is usually the best starting point
- `cosine` can sometimes produce a more organic transition

### `mask_warp_strength`
Controls how strongly the mask boundary is warped.

Useful for:
- breaking overly perfect straight seams
- reducing ruler-like edges

Use small values first.

### `mask_warp_frequency`
Controls the frequency of the warp pattern.

Lower values:
- broader, slower undulation

Higher values:
- more frequent contour variation

Use moderate values unless you specifically want stronger irregularity.

---

## ✅ Recommended starting settings

A good baseline:

- `tile_resolution = 1024`
- `padding_px = 128`
- `blend_px = 48`
- `feather_curve = smootherstep`
- `mask_warp_strength = 1.0`
- `mask_warp_frequency = 1.0`

Then adjust from there depending on:
- VRAM
- image resolution
- denoise strength
- how aggressive the enhancement is

---

## 💡 Why this approach works well

Compared with simpler overlapping-tile systems, this setup aims to reduce:

- duplicated semantic regions
- unstable multi-tile competition
- harsh seam geometry
- overly mechanical transitions

The key difference is that tiles are not treated as fully overlapping equal windows.  
They are treated as **owner regions with contextual padding**, then blended only where necessary.

---

## 📦 Installation

Clone into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lucasgattas/ComfyUI-Egregora-Divide-And-Enhance.git
```

Then restart ComfyUI.

---

## 📋 Current node list

- `Egregora Algorithm`
- `Egregora Divide Select`
- `Egregora Combine`
- `Egregora Debug Mask`

---

## 📝 Notes

- older screenshots and previews were removed because the implementation changed substantially
- some earlier experiments were intentionally removed to keep the current workflow cleaner
- if you update from an older version, recreating nodes in an existing workflow may be necessary after schema changes

---

## 🙌 Credits

Special thanks to the ideas and workflows that helped shape this project.

- **Divide and Conquer** for the broader inspiration around tiled enhancement workflows
- **comfyui-image-tiled-nodes** by **Tuki** for strong practical reference points around tiled masks and recombination

This project aims to combine strengths from both directions into a cleaner and more flexible tiled enhancement workflow for ComfyUI.

---

## 📜 License

GPL-3.0
