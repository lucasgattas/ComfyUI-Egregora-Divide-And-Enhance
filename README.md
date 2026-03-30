# ComfyUI · Egregora: Divide & Enhance 🧩✨

A focused set of ComfyUI nodes for tiled image processing, designed to split images into owner-based padded tiles, generate seam-aware masks, and recombine them with smoother, more natural transitions.

**Egregora: Divide & Enhance** is built for high-quality tiled upscaling and enhancement workflows where a large image must be processed in parts without losing spatial consistency.

## 🌟 Core features

- 🧠 deterministic tile planning  
- 🧩 owner-based base regions  
- 🖼️ padded tiles for context  
- 🎭 seam-aware mask generation  
- 🔗 adaptive recombination with owner-priority blending  
- 🌊 smoother transitions with optional mask warping  

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

- **owner-priority recombination**  
  the merge stage can favor one tile more decisively in difficult overlap regions, reducing ghosting and duplicated structure

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

**Optional inputs**
- `combine_mode`
- `dominance_gamma`
- `conflict_boost`
- `edge_boost`
- `conflict_power`
- `edge_power`
- `transition_focus`

**Outputs**
- `IMAGE`

**What it does**
- places each processed tile back into its exact padded location
- applies the provided masks directly
- supports owner-priority recombination for more decisive overlap handling
- uses adaptive local dominance to reduce ghosting when neighboring tiles disagree
- can boost ownership in stronger conflict and edge regions
- keeps the split and merge stages aligned by reusing the exact masks generated earlier

**Why this matters**
The combine stage does not rebuild masks from scratch.  
It uses the masks already generated during the divide stage, which keeps split and merge behavior aligned.

In stronger tiled enhancement workflows, this also helps reduce:
- duplicated edges
- ghosting in overlap regions
- weak 50/50 blends between semantically different tiles
- visible seam competition in thin structures like cables, metal parts, hair, text, and fine detail

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

### 5. Owner-priority recombination
The merge stage can favor one tile more strongly in difficult overlap zones instead of averaging incompatible structures together.

### 6. Adaptive local dominance
In overlap regions with stronger disagreement or stronger edges, the recombination can become more decisive, helping reduce ghosting and duplicated detail.

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

### `combine_mode`
Controls how processed tiles are recombined.

Available options:
- `owner_alpha_over`
- `normalized`

General guidance:
- `owner_alpha_over` is the recommended mode for difficult tiled enhancement workflows
- `normalized` is available for comparison and compatibility
- `owner_alpha_over` is usually better when tiles diverge more strongly in the overlap area

### `dominance_gamma`
Controls the base strength of tile ownership in the transition region.

Higher values:
- make ownership more decisive
- reduce mixing between disagreeing tiles
- can make transitions harder if pushed too far

Lower values:
- keep transitions softer
- may allow more residual overlap blending

### `conflict_boost`
Controls how much local disagreement increases ownership decisiveness.

Useful for:
- reducing ghosting
- suppressing duplicated structure in overlap zones
- making merges more assertive where tiles visibly disagree

### `edge_boost`
Controls how much local edge structure increases ownership decisiveness.

Useful for:
- cables
- hair
- metal parts
- text
- thin high-frequency detail

### `conflict_power`
Shapes how strongly local disagreement responds after normalization.

Lower values:
- make the effect more sensitive
- allow more areas to react to disagreement

Higher values:
- make the effect more selective
- emphasize only stronger disagreement zones

### `edge_power`
Shapes how strongly edge structure responds after normalization.

Lower values:
- make the edge effect broader

Higher values:
- focus the effect more on stronger edges

### `transition_focus`
Controls how concentrated the adaptive behavior is inside the transition band.

Lower values:
- spread the adaptive effect more broadly through the blend region

Higher values:
- focus the effect more tightly near the center of the seam transition

---

## ✅ Recommended starting settings

A good baseline:

- `tile_resolution = 1024`
- `padding_px = 192`
- `blend_px = 48`
- `feather_curve = smootherstep`
- `mask_warp_strength = 1.0`
- `mask_warp_frequency = 1.0`
- `combine_mode = owner_alpha_over`
- `dominance_gamma = 1.30`
- `conflict_boost = 0.90`
- `edge_boost = 0.75`
- `conflict_power = 1.00`
- `edge_power = 1.20`
- `transition_focus = 1.50`

### 📍 Practical note on `padding_px`
The default node value does not need to change, but in testing, **`padding_px = 192`** proved to be a particularly strong starting point for continuity and seam quality.

If you are using stronger denoise, more aggressive enhancement, or visually complex tiles, trying `padding_px = 192` early is recommended.

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
- overlap ghosting in high-detail regions

The key difference is that tiles are not treated as fully overlapping equal windows.  
They are treated as **owner regions with contextual padding**, then blended only where necessary, with a recombination stage that can become more decisive when tiles disagree.

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
- the current combine behavior has been tuned specifically to improve seam handling under stronger tile disagreement

---

## 🙌 Credits

Special thanks to the projects and ideas that helped shape this node pack.

- **[ComfyUI_Steudio](https://github.com/Steudio/ComfyUI_Steudio)**  
  for important inspiration around tiled enhancement workflows and broader practical experimentation in this area

- **[comfyui-image-tiled-nodes](https://github.com/tuki0918/comfyui-image-tiled-nodes)** by **tuki0918**  
  for valuable practical reference points around tiled masks, overlap handling, and tile recombination

This project aims to combine strengths from both directions into a cleaner and more flexible tiled enhancement workflow for ComfyUI.

---

## 📜 License

GPL-3.0
