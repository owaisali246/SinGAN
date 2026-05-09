# SinGAN

[Project](https://tamarott.github.io/SinGAN.htm) | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) | [CVF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf) | [Supplementary](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Shaham_SinGAN_Learning_a_ICCV_2019_supplemental.pdf) | [Talk (ICCV'19)](https://youtu.be/mdAcPe74tZI?t=3191)

*SinGAN: Learning a Generative Model from a Single Natural Image*
**ICCV 2019 Best Paper Award (Marr Prize)**

---

## Table of Contents

1. [What is SinGAN?](#what-is-singan)
2. [How SinGAN Works](#how-singan-works)
3. [Architecture Details](#architecture-details)
4. [Training Procedure](#training-procedure)
5. [Key Parameters](#key-parameters)
6. [Installation](#installation)
7. [Usage](#usage)
   - [Train](#train)
   - [Random Samples](#random-samples)
   - [Random Samples of Arbitrary Sizes](#random-samples-of-arbitrary-sizes)
   - [Animation](#animation)
   - [Harmonization](#harmonization)
   - [Editing](#editing)
   - [Paint to Image](#paint-to-image)
   - [Super Resolution](#super-resolution)
8. [Output File Structure](#output-file-structure)
9. [SIFID Benchmark](#sifid-benchmark)
10. [Practical Tips](#practical-tips)
11. [Citation](#citation)

---

## What is SinGAN?

SinGAN learns a generative model from a **single natural image** — no dataset required. Once trained, it can synthesize diverse, high-quality samples that share the same visual statistics (textures, structures, colors) as the training image, and serve as a backbone for a range of image manipulation tasks.

![Random samples from a single image](imgs/teaser.PNG)

![Application examples](imgs/manipulation.PNG)

---

## How SinGAN Works

### The Multi-Scale Pyramid

SinGAN's core idea is to learn a **pyramid of patch-GANs**, one per resolution scale. The training image is downsampled to create `N+1` scales:

```
Scale 0 (coarsest) ──► Scale 1 ──► Scale 2 ──► ... ──► Scale N (finest, original resolution)
  [tiny image]                                              [full resolution]
```

- **Scale 0** captures the overall layout and global structure (low frequency content).
- **Each subsequent scale** adds finer details and textures (higher frequency content).
- The number of scales `N` is determined automatically from the image size, `--min_size`, and `--scale_factor`.

### Generation Flow

Generation proceeds coarse-to-fine. At each scale `n`:

1. The previous scale's output `I_(n-1)` is upsampled to the current scale's resolution.
2. A noise map `z_n` (scaled by a learned amplitude `noise_amp_n`) is added to this upsampled image.
3. Generator `G_n` takes this noisy input and produces `I_n`.

Formally:
```
I_n = G_n(z_n · noise_amp_n + upsample(I_(n-1)))
```

At the coarsest scale (n=0), there is no previous image, so `I_(-1) = 0`. This means the coarsest generator must produce the full image structure from pure noise, while finer generators only need to refine details.

### Injection Scale

Applications like harmonization, editing, and paint2image work by **injecting a reference image at an intermediate scale** rather than starting from noise at scale 0. This allows:
- **Low injection scale (e.g., 1)**: The model heavily reworks the reference — global structure can change, only the finest textures are preserved.
- **High injection scale (e.g., N-1)**: The model makes subtle adjustments — the reference structure is largely preserved, only fine texture is adapted.

This injection mechanism is what gives SinGAN its versatility across tasks without retraining.

---

## Architecture Details

### Generator (`G_n`)

Each generator is a **fully convolutional network** with 5 convolutional blocks:

```
Conv(3→nfc) → BatchNorm → LeakyReLU
Conv(nfc→nfc) → BatchNorm → LeakyReLU
Conv(nfc→nfc) → BatchNorm → LeakyReLU
Conv(nfc→nfc) → BatchNorm → LeakyReLU
Conv(nfc→3) → Tanh
```

- **`nfc`**: Number of feature channels (default 32 at coarsest scale, up to `--nfc` max at finest).
- **Kernel size**: 3×3 with zero-padding, so the network is fully convolutional and resolution-agnostic.
- **Residual connection**: `G_n` outputs a residual added to its upsampled input — it only needs to learn what to *add*, not reconstruct from scratch.
- **Receptive field**: Determined by `--ker_size` (3) and `--num_layer` (5); controls how large a patch each generator "sees".

### Discriminator (`D_n`)

Each discriminator is a **PatchGAN** — fully convolutional, no sigmoid at the end (used with WGAN-GP):

```
Conv(3→nfc) → LeakyReLU
Conv(nfc→nfc) → LeakyReLU
Conv(nfc→nfc) → LeakyReLU
Conv(nfc→nfc) → LeakyReLU
Conv(nfc→1)
```

- No BatchNorm in the discriminator (standard practice for WGAN-GP).
- Outputs a heatmap of real/fake scores per patch, not a single scalar — this enforces texture realism across all spatial positions.

---

## Training Procedure

Each scale is trained **sequentially** from coarsest to finest. All lower-scale generators are frozen when training scale `n`.

### Loss Function

The total generator loss at scale `n` combines two terms:

```
L_total = L_adv + α · L_rec
```

**Adversarial loss (`L_adv`)**: WGAN-GP loss, which trains `G_n` to fool `D_n`:
```
L_adv = -E[D_n(G_n(z + upsample(I_(n-1))))]
      + λ · gradient_penalty
```

**Reconstruction loss (`L_rec`)**: Forces `G_n` to be able to exactly reconstruct the real image at this scale when given the fixed optimal noise `z*_n` (stored as `Z_opt`):
```
L_rec = ||G_n(z*_n · noise_amp_n + upsample(real_(n-1))) - real_n||²
```

The reconstruction loss anchors each generator so it can reproduce the training image exactly. The `α` parameter (`--alpha`, default 10) controls how strongly reconstruction is enforced vs. diversity.

### Noise Amplitude

The noise amplitude `noise_amp_n` for each scale is set **automatically** after the previous scale is trained:

```
noise_amp_n = sqrt(MSE(upsample(I_(n-1)), real_n)) · noise_amp_init
```

This adapts the noise injection strength to the residual reconstruction error at that scale — coarser scales (higher error) get larger amplitudes, finer scales get smaller ones.

### What Gets Saved Per Scale

After training scale `n`, the following are saved to `TrainedModels/<image_name>/scale_factor=<f>,alpha=<a>/scale<n>/`:
- `netG.pth` — generator weights
- `netD.pth` — discriminator weights
- `z_opt.pth` — the fixed reconstruction noise `z*_n`

After all scales complete, the full pyramid is saved to the root model directory:
- `Gs.pth` — list of all generators
- `Zs.pth` — list of all `z*_n` tensors
- `reals.pth` — list of real images at each scale
- `NoiseAmp.pth` — list of noise amplitudes per scale

---

## Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `--max_size` | 250 | Longest dimension of the finest scale. Higher = more detail, longer training. **Must be identical between training and all inference commands.** |
| `--min_size` | 25 | Longest dimension of the coarsest scale. Controls the total number of pyramid levels. |
| `--scale_factor` | 0.75 | Downsampling ratio between consecutive scales. 0.75 means each scale is 75% the size of the next finer one. |
| `--niter` | 2000 | Training iterations per scale. Reduce to 1000 for ~2× speedup with modest quality trade-off. |
| `--alpha` | 10 | Weight of reconstruction loss vs. adversarial loss. Higher = less diversity, more faithful reconstruction. |
| `--nfc` | 32 | Base number of feature channels in G and D. Higher = more capacity, more memory. |
| `--noise_amp` | 0.1 | Initial noise amplitude (auto-scaled per scale after the first). |
| `--not_cuda` | False | Run on CPU. Required if no GPU is available. |

---

## Installation

```bash
python -m pip install -r requirements.txt
```

**Tested with:** Python 3.6, PyTorch 1.4

**Compatibility notes for newer environments:**

- **PyTorch > 1.4**: The optimization scheme may behave differently. An alternative compatible implementation is available at [kligvasser/SinGAN](https://github.com/kligvasser/SinGAN).
- **scikit-image >= 0.19**: `morphology.binary_dilation` renamed parameter `selem` → `footprint`. Apply this fix in `SinGAN/functions.py:440`:
  ```python
  # Change:
  mask = morphology.binary_dilation(mask, selem=element)
  # To:
  mask = morphology.binary_dilation(mask, footprint=element)
  ```

---

## Usage

### Train

Place your training image under `Input/Images/`, then run:

```bash
python main_train.py --input_name <image_file_name>
```

To control output resolution (recommended for GPU training):

```bash
python main_train.py --input_name <image_file_name> --max_size 256
```

To run on CPU:

```bash
python main_train.py --input_name <image_file_name> --not_cuda
```

After training, the model is saved to:
```
TrainedModels/<image_name>/scale_factor=<f>,alpha=<n>/
```

Training also automatically generates 50 random samples at full scale, saved to:
```
Output/RandomSamples/<image_name>/gen_start_scale=0/
```

---

### Random Samples

Generate diverse samples from a trained model. The generation start scale controls diversity vs. structure:

```bash
python random_samples.py --input_name <image_file_name> \
    --mode random_samples \
    --gen_start_scale <scale_number>
```

- `--gen_start_scale 0`: Full diversity — structure and texture are both randomly generated (most varied results).
- `--gen_start_scale N-1`: Only the finest texture is randomized — structure is fixed from the training image.
- `--gen_start_scale N`: Produces near-identical outputs (no randomness at coarse scales).

**Important:** Pass the same `--max_size` value used during training.

---

### Random Samples of Arbitrary Sizes

Generate samples at a different spatial resolution than the training image:

```bash
python random_samples.py --input_name <image_file_name> \
    --mode random_samples_arbitrary_sizes \
    --scale_h <horizontal_factor> \
    --scale_v <vertical_factor>
```

- `--scale_h 1.5 --scale_v 1.0` produces images 1.5× wider than the training image.
- Factors < 1 produce smaller images; factors > 1 produce larger, tiled-texture images.

---

### Animation

Generates a short looping animation by smoothly walking through the noise space of the trained generators. This requires **re-training** with a special noise-padding mode — you cannot reuse a standard trained model:

```bash
python animation.py --input_name <image_file_name>
```

This automatically triggers a new training run with `mode=animation_train` and saves the model separately. The resulting GIF is saved to `Output/Animation/<image_name>/`.

**How it works:** At each animation frame, the noise inputs are smoothly interpolated using a momentum-based random walk (parameters `alpha`, `beta` in `generate_gif`). This produces coherent motion rather than flickering between independent random samples.

---

### Harmonization

Blends a naively pasted object into a background image so it matches the background's texture and appearance.

**Setup:**
1. Train SinGAN on the **background** image.
2. Create a naively composited image (object pasted onto background) and its binary mask.
3. Save both under `Input/Harmonization/`:
   - `<ref_name>.png` — the composited image
   - `<ref_name>_mask.png` — binary mask (white = object region)

```bash
python harmonization.py --input_name <background_image> \
    --ref_name <composited_image> \
    --harmonization_start_scale <scale>
```

- **Scale 1** (coarsest allowed): Strongest effect — color, texture, and lighting are all adapted.
- **Higher scales**: Subtler effect — only fine texture is harmonized, large-scale appearance is preserved.

**How it works:** The composited reference is downsampled and injected at the chosen scale. The SinGAN generators from that scale onward re-synthesize the image, adapting the pasted region to match the background's learned patch statistics. A dilated, Gaussian-blurred version of the mask is used to softly blend the harmonized region with the original.

Output is saved to `Output/Harmonization/<background>/<ref>_out/`.

---

### Editing

Adapts a manually edited image so that edits look consistent with the original image's texture and style.

**Setup:**
1. Train SinGAN on the **original** image.
2. Create your edited version and a binary mask marking the edited region.
3. Save both under `Input/Editing/`.

```bash
python editing.py --input_name <original_image> \
    --ref_name <edited_image> \
    --editing_start_scale <scale>
```

Outputs both a masked and unmasked version of the result. The injection scale controls how aggressively the model adapts the edit — lower scales allow more structural changes, higher scales preserve more of the edit's large-scale content.

Output is saved to `Output/Editing/<image>/<ref>_out/`.

---

### Paint to Image

Converts a coarse paint or sketch into a photorealistic image with the texture and appearance of the training image.

**Setup:**
1. Train SinGAN on the **target realistic image**.
2. Create your paint/sketch and save it under `Input/Paint/`.

```bash
python paint2image.py --input_name <realistic_image> \
    --ref_name <paint_image> \
    --paint_start_scale <scale>
```

**How it works:** The paint is injected at the specified scale. At coarser injection scales, the generators use the paint only for overall color distribution and layout, freely synthesizing realistic textures. At finer scales, the paint's details are more directly reflected in the output.

**Advanced — Quantization flag:**

```bash
python paint2image.py --input_name <realistic_image> \
    --ref_name <paint_image> \
    --paint_start_scale <scale> \
    --quantization_flag True
```

With this flag, only the injection scale's generator is re-trained on a color-quantized version of the upsampled coarse output. This can produce sharper color boundaries for some images.

Output is saved to `Output/Paint2image/<image>/<ref>_out/`.

---

### Super Resolution

Upscales a low-resolution image using a SinGAN model trained on the LR image itself:

```bash
python SR.py --input_name <LR_image_file_name>
```

Default upsampling factor is 4×. For a different factor:

```bash
python SR.py --input_name <LR_image_file_name> --sr_factor <factor>
```

**How it works:** A SinGAN model is trained with `mode=SR_train`. The scale pyramid is set up so the finest scale corresponds to the target HR resolution. Unlike external-dataset SR methods, SinGAN exploits the internal patch recurrence within the single LR image — it finds repeated patches at different scales and uses them to hallucinate high-frequency details.

SR results on the BSD100 dataset are available in the Downloads folder.

---

## Output File Structure

```
SinGAN/
├── Input/
│   ├── Images/              # Training images (.png, .jpg)
│   ├── Harmonization/       # Composited images + masks for harmonization
│   ├── Editing/             # Edited images + masks for editing
│   └── Paint/               # Paint/sketch images for paint2image
│
├── TrainedModels/
│   └── <image_name>/
│       └── scale_factor=<f>,alpha=<n>/   # One folder per training run
│           ├── Gs.pth                    # All generators (list)
│           ├── Zs.pth                    # All z_opt tensors (list)
│           ├── reals.pth                 # All real images at each scale (list)
│           ├── NoiseAmp.pth              # All noise amplitudes (list)
│           └── scale0/, scale1/, ...     # Per-scale checkpoints
│               ├── netG.pth
│               ├── netD.pth
│               └── z_opt.pth
│
└── Output/
    ├── RandomSamples/<image_name>/gen_start_scale=<n>/   # 0.png, 1.png, ...
    ├── RandomSamples_ArbitrarySizes/<image_name>/
    ├── Animation/<image_name>/
    ├── Harmonization/<background>/<ref>_out/
    ├── Editing/<image>/<ref>_out/
    ├── Paint2image/<image>/<ref>_out/
    └── SR/<sr_factor>/
```

---

## SIFID Benchmark

**Single Image Fréchet Inception Distance (SIFID)** measures the quality of generated samples by comparing the distribution of deep CNN features (from InceptionV3's `conv3_3` layer) between real patches and generated patches. Lower is better.

Unlike standard FID (which requires thousands of images), SIFID operates on a single real image vs. a set of generated samples, making it appropriate for evaluating single-image generative models.

### Running SIFID

```bash
python SIFID/sifid_score.py \
    --path2real <folder_containing_real_images> \
    --path2fake <folder_containing_generated_images> \
    --images_suffix png
```

### Setting Up the Real Image Folder

SIFID pairs files by sorted filename index. To evaluate against 50 generated samples, copy the real image 50 times with numeric names:

```bash
mkdir -p SIFID/real_images/<image_name>
for i in $(seq 0 49); do
    cp Input/Images/<image_name>.png SIFID/real_images/<image_name>/$i.png
done
```

Then point `--path2real` at that folder and `--path2fake` at the generated samples folder (e.g., `Output/RandomSamples/<image_name>/gen_start_scale=0`).

### Interpreting Results

| SIFID Score | Interpretation |
|---|---|
| < 0.00005 | Excellent — generated textures closely match the real image |
| 0.00005 – 0.0002 | Good — visually plausible, minor distributional differences |
| > 0.0005 | Poor — generated samples look noticeably different from the real image |

Note: SIFID measures texture-level statistical similarity, not perceptual quality. A very low SIFID can still produce images with structural artifacts; always visually inspect results alongside the metric.

---

## Practical Tips

### Consistency Between Training and Inference

`--max_size` determines the finest scale resolution. It **must be identical** between training and every inference command that loads that model. Mismatches cause tensor size errors during generation. If in doubt, check the training command used or infer from the saved `reals.pth`:

```python
import torch
reals = torch.load('TrainedModels/<name>/scale_factor=0.750000,alpha=10/reals.pth', map_location='cpu')
print(reals[-1].shape)  # finest scale shape
```

### Recommended Settings for GPU Training (e.g., T4)

```bash
python main_train.py --input_name <image> --max_size 256 --niter 1000
```

- `--max_size 256`: Keeps the finest scale at 256px on the longest dimension.
- `--niter 1000`: Halves training time with acceptable quality trade-off.
- Typical VRAM usage: ~0.17 GB per training process; System RAM is usually the limiting factor (~1.1 GB per process).

### Parallel Training

To train multiple images simultaneously, launch separate processes with a short stagger to avoid CUDA initialization conflicts:

```python
import subprocess, time
images = ['image1.png', 'image2.png', 'image3.png']
procs = []
for img in images:
    p = subprocess.Popen(['python', 'main_train.py', '--input_name', img, '--max_size', '256'])
    procs.append(p)
    time.sleep(10)  # stagger to avoid CUDA race
for p in procs:
    p.wait()
```

On a 12.7 GB RAM machine, limit to 5–6 parallel processes to avoid OOM.

### Animation Requires Separate Training

Animation cannot reuse a standard trained model. Running `animation.py` always triggers a fresh training run in `animation_train` mode, which uses noise padding instead of zero padding to enable seamless tiling. Plan for this extra training time if animation is needed.

### Images Must Have a Minimum Size

SinGAN requires the training image to be larger than `--min_size` (default 25px). For `--max_size 256`, any image ≥ 256px on its longest dimension will be downsampled to exactly 256px; smaller images will be used as-is (down to 25px minimum at the coarsest scale).

---

## Citation

```bibtex
@inproceedings{rottshaham2019singan,
  title={SinGAN: Learning a Generative Model from a Single Natural Image},
  author={Rott Shaham, Tamar and Dekel, Tali and Michaeli, Tomer},
  booktitle={Computer Vision (ICCV), IEEE International Conference on},
  year={2019}
}
```
