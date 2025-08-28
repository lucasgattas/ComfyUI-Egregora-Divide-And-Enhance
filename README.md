<center>

# **ComfyUI · Egregora: Divide & Enhance** 🧩🚀

**⚡A set of ComfyUI nodes inspired by Divide & Conquer algorithm, designed to split, enhance, and recombine images for high-quality upscaling, with prompt mixing and analysis tools for cleaner, sharper results⚡**

</center>

---

## ✨ What is this?

**Egregora** is a small suite of custom nodes that help you **split, enhance, and recombine** images—plus a clean **two‑slider SDXL prompt mixer** that keeps things simple while staying robust for low‑step samplers.

Inspired by **Steudio’s Divide & Conquer** nodes, adapted and refactored to fit a streamlined upscaling workflow. 🧠✂️🧵

---

## 🧱 Included Nodes

### 🚀 Egregora Turbo Prompt

Two prompts, two sliders—done, one for captioning (Florence2 or others), and one for Global positive and negative prompts. Builds **SDXL‑ready CONDITIONING** with:

* auto‑read **size** from LATENT (ADM: width/height/target/crop),
* proper **pooled\_output** handling (1280‑d),
* **combine‑style** mixing (cleaner than embedding averages),
* optional **negative** and a simple **blacklist** (remove words from caption).

### 🧠 Egregora Algorithm

Planner for Divide‑and‑Conquer upscaling. Computes tile layout given target size & overlap; supports tile ordering strategies and sensible defaults to avoid seams.

### ✂️ Egregora Divide & Select

Splits an image/latent according to the planner. You can pass all tiles, or pick specific ones for targeted enhancement.

### 🔗 Egregora Combine

Merges processed tiles back into a seamless image using robust blending (distance‑field / multi‑scale style approaches to avoid borders and repetition).

### 👁️ Egregora Preview

Visual overlay to preview the tile grid, overlaps, and order—verify the plan before you spend compute.

### 🔍 Egregora Content Analysis *(optional)*

Analyzes complexity/detail to hint at better tile sizes or overlaps for tricky images.

---

## 📦 Installation

1. Clone into ComfyUI’s custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lucasgattas/comfyui-egregora-divide-and-enhance.git
```

2. Ensure the folder contains `__init__.py` and `egregora_divide_and_enhance.py`.
3. Restart ComfyUI. Nodes appear under **Egregora/**.

> **Note:** You can rename the Python file; the package `__init__.py` controls what gets imported.

---

## 🛠️ Minimal Workflow (SDXL)

**Goal:** Use Turbo Prompt for text conditioning and keep the rest of your graph simple.

1. **Load Model (SDXL)** → **CLIP**
2. **Egregora Turbo Prompt**

   * *caption\_text* = your main description
   * *global\_positive\_prompt* = secondary style or theme
   * *caption\_strength* / *global\_strength* = only two sliders you need
   * (wire **latent** here so ADM size auto‑matches your sampler resolution)
3. **KSampler** (Euler/Karras or your favorite)

   * Positive/Negative from Turbo Prompt, model & latent as usual.
4. **VAE Decode** → **Preview / Save**

**Upscaling (Divide & Enhance):**

* Use **Algorithm → Divide & Select → \[process tiles] → Combine**.
  Start with moderate tile sizes and overlap; preview first.

---

## 🎚️ Practical Tips

* Keep prompts **concise**; long texts dilute signal.
* For realism vs. cartoon conflicts, add targeted **negatives** (e.g., `cartoon, plush, chibi`) so the portrait holds shape while the global style adds texture/color.
* Turbo Prompt avoids fragile scheduling and mixes prompts via multiple conditionings (sampler‑level combine).

---

## 🧪 Troubleshooting

* **Flat/grey outputs** → raise steps slightly or increase the stronger slider; ensure the CLIP/model are SDXL compatible.
* **Repetitions/Grid looks** → check that you aren’t applying external tiling patches unintentionally; reduce extreme overlaps or tile size if using the Divide/Combine flow.
* **Shape/key errors** → update to the latest version; Turbo Prompt sets `pooled_output` and ADM keys; make sure you pass a valid LATENT.

---

## 🧾 Folder Structure

```
comfyui-egregora-divide-and-enhance/
├─ __init__.py
├─ egregora_divide_and_enhance.py
├─ README.md  ← you are here
└─ Images/    ← optional screenshots
```

---

## 🙌 Credits

* Inspired by **Steudio’s Divide & Conquer** node suite and community best practices around SDXL conditioning and tiling.
* Built for clarity: minimal controls, sensible defaults, strong integration with ComfyUI.

---

## 📜 License

GNU GENERAL PUBLIC LICENSE Version 3, see [LICENSE](LICENSE)

---

### Changelog

* **0.1.0** — Initial release: Turbo Prompt with latent‑sized ADM; Divide/Select/Combine/Preview/Analysis nodes.
