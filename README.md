# ComfyUI · Egregora: Divide & Enhance

A focused set of ComfyUI nodes for tiled image enhancement and seamless recombination.

**Egregora: Divide & Enhance** combines the practical tiling workflow of divide-and-process pipelines with a cleaner mask system for smoother, more natural merges. The current version is centered on a simple, robust node flow:

- plan the upscale and tile layout
- split into tiles
- generate masks that match the tile layout
- process the tiles however you want
- recombine them with consistent blending

This makes the pack especially useful for:
- tiled upscaling
- detail enhancement on very large images
- workflows that exceed normal VRAM limits
- controlled image processing tile by tile
- reducing visible seams when recombining processed tiles

---

## What this node pack is for

These nodes are designed for workflows where a full image is too large or too expensive to process in one pass.

Typical use cases:
- upscale a large image in manageable tiles
- send each tile through an enhancement or img2img workflow
- preserve consistent tile placement from split to merge
- inspect masks directly when debugging seams
- use feathered masks to blend tiles more naturally than hard-edge recomposition

The current implementation is especially strong when you want:
- deterministic tile placement
- mask-aware tiled recombination
- a workflow that is simple to read and easy to debug
- better seam behavior than a basic hard tile merge

---

## Included nodes

### Egregora Algorithm

Builds the tiling plan and the upscaled working canvas.

**Inputs**
- `image`
- `tile_resolution`
- `min_overlap`
- `min_scale_factor`
- `tile_order`
- `scaling_method`

**Outputs**
- `IMAGE`
- `EGREGORA_DATA`

**What it does**
- calculates the target working resolution based on `min_scale_factor`
- rescales the source image to the working canvas
- computes tile placement and overlap
- stores the plan in `EGREGORA_DATA` for downstream nodes

**Why it matters**
- the same tile plan is reused downstream
- tile placement stays deterministic
- split and combine stay aligned

---

### Egregora Divide Select

Splits the image into tiles and generates the corresponding masks.

**Inputs**
- `image`
- `egregora_data`
- `tile`
- `feather_ratio`
- `feather_curve`

**Outputs**
- `TILE(S)`
- `MASK(S)`

**What it does**
- outputs all tiles or a selected tile
- generates masks that match the exact tile layout
- applies feathering directly in the mask generation step

**Important behavior**
- `tile = 0` returns all tiles and all masks
- any positive tile index returns only the selected tile and its mask

This is useful both for production workflows and for debugging one tile at a time.

---

### Egregora Combine

Recombines processed tiles using the masks generated in the split stage.

**Inputs**
- `tiles`
- `masks`
- `egregora_data`
- `scaling_method`

**Outputs**
- `IMAGE`

**What it does**
- places each processed tile back in the correct position
- uses the provided masks directly instead of rebuilding them
- accumulates image values and mask weights
- normalizes the final result to avoid hard seams

**Why this is important**
Using the masks generated during the divide step helps keep split and merge behavior consistent. This greatly improves seam quality compared with recomputing approximate masks during recombination.

---

### Egregora Debug Mask

Outputs the generated masks directly, in the same style as tiled mask outputs used in practical ComfyUI workflows.

**Inputs**
- `egregora_data`
- `feather_ratio`
- `feather_curve`
- `tile_index`

**Outputs**
- `MASK`

**What it does**
- previews the actual masks used by the workflow
- helps diagnose seam behavior before running expensive tile processing
- makes it easy to compare feather settings and curves

**Important behavior**
- `tile_index = 0` returns the full batch of masks
- any positive tile index returns one selected mask

This node is useful for checking whether the mask itself is correct before investigating model-side tile differences.

---

## Node flow

The core workflow is now intentionally simple:

```text
Egregora Algorithm
    ↓
Egregora Divide Select
    ↓
[process tiles with your own workflow]
    ↓
Egregora Combine
```

For mask inspection:

```text
Egregora Algorithm
    ↓
Egregora Debug Mask
```

---

## Feather system

The current mask system supports:

- `feather_ratio`
- `feather_curve`

### feather_ratio
Controls how much of the overlap is used for the feather transition.

### feather_curve
Controls how the transition behaves across the feathered region.

Available curves:
- `linear`
- `smoothstep`
- `smootherstep`
- `cosine`

In practice:
- `linear` is the most direct and predictable
- `smoothstep` is softer
- `smootherstep` is usually the most natural-looking option
- `cosine` can also give a smooth blend, depending on the image

If your goal is the cleanest natural merge, `smootherstep` is often a strong starting point.

---

## How to use

### Basic tiled enhancement workflow

1. Load your source image.
2. Send it into **Egregora Algorithm**.
3. Choose:
   - a working `tile_resolution`
   - a `min_overlap`
   - a `min_scale_factor`
   - a `tile_order`
4. Send the image and `EGREGORA_DATA` into **Egregora Divide Select**.
5. Choose:
   - `tile = 0` to output all tiles
   - `feather_ratio`
   - `feather_curve`
6. Process the output tiles with your preferred enhancement pipeline.
7. Send the processed tiles, masks, and `EGREGORA_DATA` into **Egregora Combine**.
8. Save the recombined image.

---

## Practical guidance

### Choosing tile resolution
Larger tiles:
- reduce the number of tile boundaries
- may improve global consistency
- require more VRAM

Smaller tiles:
- are easier on memory
- may produce more visible variation between tiles
- increase the importance of a good overlap and mask

### Choosing overlap
More overlap:
- gives the blend more room to transition
- usually improves seam quality
- increases compute and redundancy

Too little overlap:
- makes seams harder to hide

### Choosing feather settings
Good starting point:
- `feather_ratio = 0.5`
- `feather_curve = smootherstep`

If you want a more direct mask:
- use `linear`

If you want a softer transition:
- use `smoothstep` or `smootherstep`

---

## Why use these nodes instead of a simpler tile split/merge setup?

Because the point is not only to split an image.

The useful part is having:
- a planned upscale canvas
- deterministic tile placement
- masks that match the split stage
- a clean debug path for mask inspection
- a smoother and more natural recombination result

This makes the node pack well suited for users who want more control over tiled enhancement while still keeping the graph readable.

---

## Installation

Clone into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lucasgattas/ComfyUI-Egregora-Divide-And-Enhance.git
```

Then restart ComfyUI.

---

## Current node list

The current core nodes are:

- `Egregora Algorithm`
- `Egregora Divide Select`
- `Egregora Combine`
- `Egregora Debug Mask`

---

## Notes

- older screenshots and previews were removed because the node behavior and layout changed
- the current version is focused on the actual production workflow rather than legacy previews
- if you update from an older version, recreating the nodes in an existing workflow may be necessary after schema changes

---

## Credits

Special thanks to the ideas and workflows that helped shape this project.

- **Divide and Conquer** for the broader inspiration around tiled enhancement workflows
- **comfyui-image-tiled-nodes** by **Tuki** for helping establish a strong practical reference for tile masks and recombination behavior

This project now aims to combine the strengths of both approaches into a cleaner and more flexible tiled enhancement workflow for ComfyUI.

---

## License

GPL-3.0
