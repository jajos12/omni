# OmniSub2026 — Visual Speech Recognition: From Zero to Rank #1

> **Competition deadline: March 19, 9:00 PM**  
> **Resources: 2 × RTX 3090 24 GB (48 GB total VRAM)**  
> **Task: Silent video → English transcript, evaluated by WER (lower = better)**

---

## Table of Contents

1. [Problem Formulation (Rigorous)](#1-problem-formulation-rigorous)
2. [2026 SOTA Landscape](#2-2026-sota-landscape)
3. [Architecture Decision](#3-architecture-decision)
4. [Environment Setup](#4-environment-setup)
5. [Data Preprocessing — Lip ROI Extraction](#5-data-preprocessing--lip-roi-extraction)
6. [Model Architecture (Full Code)](#6-model-architecture-full-code)
7. [Training — DDP on 2×3090](#7-training--ddp-on-23090)
8. [Language Model Rescoring (KenLM + Beam Search)](#8-language-model-rescoring-kenlm--beam-search)
9. [Inference and Submission Pipeline](#9-inference-and-submission-pipeline)
10. [WER Optimization Tactics](#10-wer-optimization-tactics)
11. [Day-by-Day Action Plan](#11-day-by-day-action-plan)
12. [Debugging Guide](#12-debugging-guide)
13. [Competition Report Template](#13-competition-report-template)

---

## 1. Problem Formulation (Rigorous)

### 1.1 What Is Being Asked

You are given a video clip $v \in \mathbb{R}^{T \times H \times W \times 3}$ of a speaking face, where $T$ is the number of frames, and $H, W$ are the spatial dimensions. There is **no audio channel** — the task is purely visual.

You must produce a transcription $\hat{y} = (w_1, w_2, \ldots, w_N)$ that minimizes Word Error Rate:

$$\text{WER} = \frac{S + D + I}{N}$$

where $S$ = substitutions, $D$ = deletions, $I$ = insertions, and $N$ = total reference words. This is the **word-level edit distance** normalized by the reference length. A score of 0.0 is perfect; 1.0 means you got nothing right; scores above 1.0 are possible (if you insert vastly more words than exist).

### 1.2 Why Is This Hard?

The core difficulty is **viseme ambiguity**. A viseme is the visual equivalent of a phoneme — it is the shape your mouth makes when producing a sound. The critical problem: many phonemes share the same viseme.

| Phoneme Set | Visual Appearance |
|-------------|-------------------|
| /p/, /b/, /m/ | Pressed lips |
| /f/, /v/ | Upper teeth + lower lip |
| /d/, /t/, /n/ | Tongue tip at alveolar ridge (invisible) |
| /k/, /g/ | Back of tongue at velum (completely invisible) |

This means `"bat"`, `"pat"`, and `"mat"` are **visually identical** in isolation. Only context disambiguates them. This is why a language model is not optional — it is part of the model's core decoding machinery.

### 1.3 Evaluation Metric — WER in Practice

Key implications you must internalize:

- **Predicting nothing** → WER = 1.0 (100% deletions). Empty string is the worst possible output.
- **Word order matters doubly** — getting the right words in the wrong order counts as both deletions *and* insertions.
- **Short clips with long ground truth** → even one missed word is penalized heavily.
- **The metric is case-insensitive and whitespace-normalized** per competition rules, so don't waste effort on casing.

The total WER over the test set is computed over the concatenated edit distance, not as an average per clip. This means longer clips with more words have proportionally more impact.

---

## 2. 2026 SOTA Landscape

### 2.1 Where the Field Stands

The field has progressed dramatically over the last 4 years. On the LRS3 benchmark (the canonical English VSR test set from TED talks):

| Model | Year | Training Data (hrs) | VSR WER (%) | Open Source |
|-------|------|---------------------|-------------|-------------|
| ResNet + LSTM (baseline era) | 2018 | 433 | 58.9 | Yes |
| TM-CTC | 2020 | 433 | 43.2 | No |
| AV-HuBERT Large | 2022 | 1,759 unlabeled | 26.9 | Yes |
| AutoAVSR (Ma et al.) | 2023 | 1,307 auto-labeled | **19.1** | **Yes** |
| LP Conformer (Google, private) | 2023 | large private | 12.8 | No |
| VALLR (Thomas et al., ICCV 2025) | 2025 | 30 hrs only | **18.7** | Partial |
| Omni-AVSR / MMS-LLaMA era | 2025 | large, LLM-based | ~15 | No |

**The key takeaway**: AutoAVSR at 19.1% WER is the best *fully reproducible, open-source* model available. VALLR achieves 18.7% with a novel phoneme-centric approach using only 30 hours of labeled data — we'll discuss this as a strategic upgrade.

### 2.2 Why AutoAVSR Is Your Base

AutoAVSR (`mpc001/Visual_Speech_Recognition_for_Multiple_Languages`) is the practical choice because:

1. It has **pretrained weights publicly available** trained on 1,307 hours.
2. The architecture is well-understood: 3D-ResNet-18 frontend + Conformer encoder + Transformer decoder, joint CTC/attention loss.
3. The ROI extraction pipeline is included and battle-tested.
4. It achieves 19.1% WER on LRS3 **visual-only** — directly comparable to this competition.

### 2.3 The VALLR Upgrade Opportunity

VALLR (ICCV 2025) proposes a two-stage approach that's conceptually cleaner and achieves better WER with *far less data*:

1. **Stage 1**: A Video Transformer with CTC head predicts **phoneme sequences** (not characters or words directly) from lip frames. There are only ~40 English phonemes — a much smaller, cleaner output space than characters.
2. **Stage 2**: The phoneme sequence is fed to a fine-tuned **LLM** (e.g. LLaMA 3.2-3B) that reconstructs coherent words and sentences using broad linguistic context.

The insight: phonemes are a better intermediate representation than characters because:
- The LLM gets a structured linguistic signal, not raw visual noise
- The first-stage network has a simpler classification problem (40 classes, not 26+ chars)
- The LLM's massive language knowledge is fully leveraged for context completion

**Strategy given our time budget**: Start with AutoAVSR (proven baseline), fine-tune, add KenLM. If WER after Day 2 is above ~40%, consider investing time in the VALLR two-stage approach on Day 3.

### 2.4 The E-Branchformer Upgrade

Competition winners at CNVSRC 2023/2024 replaced the Conformer encoder with **E-Branchformer** and consistently saw WER improvements. The architecture runs two branches in **parallel** (self-attention for global context; depthwise convolution with gated MLP for local context) and merges them, unlike the Conformer's sequential arrangement. This is a drop-in replacement with the same parameter count that costs almost nothing to switch.

**We will use E-Branchformer as our encoder instead of Conformer.**

---

## 3. Architecture Decision

### 3.1 Final Chosen Architecture

```
Input: [T × 1 × 96 × 96] — stabilized grayscale lip frames
    │
    ▼
[3D-ResNet-18 Frontend]
  Conv3d(1, 64, k=(5,7,7), s=(1,2,2)) → BN → ReLU → MaxPool3d
  → ResNet-18 stages 1–4 (applied per-frame after temporal front-end)
  Output: [T, 512] — visual feature per frame
    │
    ▼
[Linear Projection] 512 → 256
    │
    ▼
[E-Branchformer Encoder × 18 layers]
  Each block (parallel branches):
    ├── Branch A: Multi-head Self-Attention (global context)
    └── Branch B: DepthwiseConv + cgMLP (local context)
  → Enhanced merge: concat → DepthwiseConv → Linear
  Output: [T, 256] — contextualized visual features
    │
    ├─── [CTC Head]  Linear(256, vocab_size) → log_softmax
    │     Used for: training regularization + beam search initialization
    │
    └─── [Transformer Decoder × 6 layers]
          Cross-attention over encoder output
          Autoregressive character generation
          Output: [N, vocab_size] — character logits
```

**Loss**: `L = 0.3 × CTC_loss + 0.7 × CE_loss`

The 0.3/0.7 split is standard from the AutoAVSR paper. CTC encourages the encoder to produce monotonic, aligned representations; attention decoder handles context and long-range corrections.

### 3.2 Vocabulary Design

Use character-level vocabulary (not subword/BPE) — simpler, more robust for fine-tuning on small datasets:

```
Index 0: <blank>   (CTC blank token)
Index 1: <sos>     (start of sequence, for attention decoder)
Index 2: <eos>     (end of sequence)
Index 3–28: a–z    (26 lowercase letters)
Index 29: ' '      (space)
Total vocab size: 30
```

### 3.3 Pretrained Weights Strategy

Do **not** train from scratch. The weights from AutoAVSR encode 1,307 hours of lip-reading knowledge. Fine-tuning proceeds as follows:

| Component | Learning Rate | Why |
|-----------|--------------|-----|
| 3D-ResNet frontend | `1e-5` | Low LR — visual features are universal, don't destroy them |
| E-Branchformer encoder | `1e-4` | Can adapt more freely |
| CTC head | `3e-4` | Needs to adapt to new vocabulary distribution |
| Attention decoder | `3e-4` | Most domain-specific — can change significantly |

Use layer-wise LR decay with decay factor 0.8 per layer group going backward.

---

## 4. Environment Setup

### 4.1 System Requirements

- Ubuntu 20.04 / 22.04
- CUDA 12.1+
- Python 3.10
- Both GPUs visible: `nvidia-smi` should show two RTX 3090s

### 4.2 Step-by-Step Installation

```bash
# Create isolated environment
conda create -y -n vsr python=3.10
conda activate vsr

# PyTorch with CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Core VSR dependencies
pip install mediapipe==0.10.9
pip install opencv-python-headless==4.9.0.80
pip install einops==0.7.0
pip install omegaconf==2.3.0
pip install hydra-core==1.3.2
pip install pandas==2.2.0
pip install tqdm==4.66.2

# Audio/language model dependencies  
pip install kenlm==0.2.0
pip install pyctcdecode==0.5.0  # Beam search decoder with KenLM integration

# Distributed training
pip install deepspeed==0.14.0   # Optional but useful for memory optimization
pip install lightning==2.2.0    # Optional — makes DDP boilerplate cleaner

# For KenLM compilation (needed to build n-gram LMs)
sudo apt-get install -y libboost-all-dev cmake

# Clone AutoAVSR — this is your model zoo + ROI extraction pipeline
git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
cd Visual_Speech_Recognition_for_Multiple_Languages
pip install -r requirements.txt
cd ..

# Download pretrained VSR weights (LRS3 visual-only model)
# From the repo's README, download:
# VSR model: LRS3, WER: 19.1%
# Filename: model_vsr_lrs3vox_base_s3fd.pth
# Place in: ./checkpoints/pretrained/
mkdir -p checkpoints/pretrained
# (Download from the URL listed in mpc001 repo releases page)
```

### 4.3 Verify Multi-GPU Setup

```python
# verify_gpus.py
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

# Expected output:
# CUDA available: True
# GPU count: 2
#   GPU 0: NVIDIA GeForce RTX 3090, 24.0 GB
#   GPU 1: NVIDIA GeForce RTX 3090, 24.0 GB
```

### 4.4 Project Directory Structure

```
omnisub2026/
├── data/
│   ├── train/
│   │   └── <video_id>/<clip_id>.mp4
│   │   └── <video_id>/<clip_id>.txt
│   ├── test/
│   │   └── <video_id>/<clip_id>.mp4
│   └── rois/          ← pre-extracted ROI .npy files (you create this)
│       ├── train/
│       │   └── <video_id>/<clip_id>.npy
│       └── test/
│           └── <video_id>/<clip_id>.npy
├── checkpoints/
│   ├── pretrained/
│   │   └── model_vsr_lrs3vox_base_s3fd.pth
│   └── finetuned/
│       └── epoch_XX.pt
├── lm/
│   ├── librispeech_lm_4gram.arpa   ← download + build
│   └── librispeech_lm_4gram.bin
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── sample_submission.csv
└── submission.csv
```

---

## 5. Data Preprocessing — Lip ROI Extraction

This is the most fragile and most important step. Bad ROI = everything downstream fails. Do it once, save to disk, never repeat during training.

### 5.1 The Affine Stabilization Math

**Why affine and not just bounding box?** A bounding box crop wobbles as the head moves — every frame has slightly different scale, rotation, and position. The downstream model would waste capacity compensating for head motion rather than reading lips.

An affine transform $M \in \mathbb{R}^{2 \times 3}$ maps any 2D point $\mathbf{p} = [x, y, 1]^T$ to its stabilized position:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = M \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

It encodes a **similarity transform** (rotation + uniform scale + translation — 4 degrees of freedom). This is more constrained than the full affine (6 DOF), which is intentional: we don't want shear, and uniform scale keeps the face proportions intact.

**The canonical template**: We define fixed target pixel positions for the two eye centers in the output 96×96 image. These come from aligning thousands of frontal faces and averaging the eye positions:

```
Left eye center  → (38.2946, 51.6963)  in 96×96 image
Right eye center → (73.5318, 51.5014)  in 96×96 image
```

Given detected eye positions $\mathbf{e}_L, \mathbf{e}_R$ in the original frame, we solve:

$$M = \arg\min_M \|M[\mathbf{e}_L; 1] - \mathbf{c}_L\|^2 + \|M[\mathbf{e}_R; 1] - \mathbf{c}_R\|^2$$

`cv2.estimateAffinePartial2D` solves this exactly (closed form for 2-point similarity).

**Why use eyes for alignment, not the mouth directly?** The eyes are more stable anatomically and provide a longer baseline (wider apart), making the transform more numerically stable. Once eyes are locked, the mouth is at a predictable canonical location.

### 5.2 Full ROI Extractor (`src/preprocess.py`)

```python
"""
src/preprocess.py

Run ONCE before any training:
    python src/preprocess.py --data_root data/ --output_root data/rois/

Extracts and saves stabilized 96×96 grayscale lip ROI tensors as .npy files.
MediaPipe runs on CPU — this is a CPU-bound preprocessing job.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import argparse
import json

# ─── Canonical eye positions in 96×96 output image ────────────────────────────
CANONICAL_EYES = np.array([
    [38.2946, 51.6963],   # left eye center
    [73.5318, 51.5014],   # right eye center
], dtype=np.float32)

# Approximate canonical mouth center in the 4× canvas (used for final crop)
# Derived from: eyes at y≈52 in 96×96, mouth is typically ~20px below midpoint
CANVAS_MULT = 4          # We warp to 4× canvas then crop
MOUTH_CANONICAL_Y = CANVAS_MULT * 48 + 19   # ≈ 211 in 384×384 canvas
MOUTH_CANONICAL_X = CANVAS_MULT * 48        # ≈ 192, horizontally centered

# MediaPipe landmark indices
LEFT_EYE_INNER   = 133   # inner canthus
LEFT_EYE_OUTER   = 33    # outer canthus
RIGHT_EYE_INNER  = 362
RIGHT_EYE_OUTER  = 263

TARGET_SIZE = 96


class LipROIExtractor:
    """
    Converts an .mp4 video to a stabilized sequence of 96×96 grayscale lip frames.
    
    Handles:
      - MediaPipe detection failures (uses last-known-good transform)
      - Very short clips (<5 frames)
      - Large head rotations (MediaPipe is robust to ~45°)
      - Low-resolution inputs (upsampling fallback)
    """

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,    # Video mode: uses tracking between frames
            max_num_faces=1,
            refine_landmarks=True,      # 478 landmarks including iris + lip detail
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._last_M: np.ndarray | None = None   # Cache of last valid affine matrix
        self._fail_count = 0

    def _landmark_px(self, lm, idx: int, w: int, h: int) -> np.ndarray:
        """Convert normalized MediaPipe landmark to pixel coordinates."""
        return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

    def _compute_affine(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Run face mesh on frame, extract eye landmarks, compute affine matrix.
        Returns 2×3 affine matrix, or None on detection failure.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        left_eye  = (self._landmark_px(lm, LEFT_EYE_INNER,  w, h) +
                     self._landmark_px(lm, LEFT_EYE_OUTER,  w, h)) / 2
        right_eye = (self._landmark_px(lm, RIGHT_EYE_INNER, w, h) +
                     self._landmark_px(lm, RIGHT_EYE_OUTER, w, h)) / 2

        src = np.array([left_eye, right_eye], dtype=np.float32)

        # estimateAffinePartial2D: similarity transform (rotation+scale+translation)
        # Full affine (estimateAffine2D) would also fit shear — we don't want that.
        M, inliers = cv2.estimateAffinePartial2D(
            src, CANONICAL_EYES,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        return M  # Shape (2, 3) or None

    def _apply_crop(self, frame: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Apply affine warp to full frame, then crop the mouth region.
        Returns (TARGET_SIZE, TARGET_SIZE) uint8 grayscale image.
        """
        canvas_size = TARGET_SIZE * CANVAS_MULT   # 384 × 384

        warped = cv2.warpAffine(
            frame, M,
            (canvas_size, canvas_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,   # Replicate edges — avoids black borders
        )

        # Crop TARGET_SIZE × TARGET_SIZE centered on canonical mouth location
        half = TARGET_SIZE // 2
        y1 = MOUTH_CANONICAL_Y - half
        y2 = MOUTH_CANONICAL_Y + half
        x1 = MOUTH_CANONICAL_X - half
        x2 = MOUTH_CANONICAL_X + half

        # Safety clamp (shouldn't trigger for normal faces)
        y1, y2 = max(0, y1), min(canvas_size, y2)
        x1, x2 = max(0, x1), min(canvas_size, x2)

        crop = warped[y1:y2, x1:x2]

        # Resize to TARGET_SIZE if clamping changed the shape
        if crop.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
            crop = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

        # Convert to grayscale — VSR models use single channel
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        return gray  # dtype: uint8, shape: (96, 96)

    def extract(self, video_path: str) -> np.ndarray:
        """
        Extract stabilized lip ROI sequence from video.
        
        Returns:
            np.ndarray of shape (T, 96, 96), dtype float16, values in [0, 1]
        Raises:
            ValueError if no valid frames could be extracted.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames = []
        self._last_M = None
        self._fail_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            M = self._compute_affine(frame)

            if M is None:
                self._fail_count += 1
                # Strategy: try upsampling the frame (helps with low-res inputs)
                if self._fail_count <= 3:
                    up = cv2.resize(frame, None, fx=2.0, fy=2.0,
                                    interpolation=cv2.INTER_CUBIC)
                    M_up = self._compute_affine(up)
                    if M_up is not None:
                        # M_up maps from 2× space; scale down by 0.5
                        scale = np.diag([0.5, 0.5, 1.0])[:2, :]
                        M = M_up @ np.vstack([np.eye(2, 3), [0, 0, 1]])
                        # Simplified: just scale the translation component
                        M_fixed = M_up.copy()
                        M_fixed[:, 2] *= 0.5
                        M = M_fixed

                if M is None and self._last_M is not None:
                    # Fall back to last-known-good transform
                    M = self._last_M
                elif M is None:
                    # First frame failed and no fallback — append blank
                    blank = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
                    frames.append(blank)
                    continue

            self._last_M = M
            crop = self._apply_crop(frame, M)
            frames.append(crop)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Zero frames extracted from {video_path}")

        # Stack and normalize
        arr = np.stack(frames, axis=0).astype(np.float16) / 255.0  # (T, 96, 96)
        return arr

    def close(self):
        self.face_mesh.close()


def preprocess_split(data_root: str, output_root: str, split: str, skip_existing: bool = True):
    """
    Process all .mp4 files in data_root/split/ and save .npy to output_root/split/.
    """
    data_path = Path(data_root) / split
    out_path  = Path(output_root) / split

    videos = sorted(data_path.rglob("*.mp4"))
    print(f"\n[{split.upper()}] Found {len(videos)} videos")

    extractor = LipROIExtractor()
    stats = {"success": 0, "failed": 0, "skipped": 0}

    for video_path in tqdm(videos, desc=f"Extracting {split}"):
        # Compute output path (mirrors input structure)
        rel_path = video_path.relative_to(data_path)
        npy_path = out_path / rel_path.with_suffix(".npy")
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and npy_path.exists():
            stats["skipped"] += 1
            continue

        try:
            roi = extractor.extract(str(video_path))  # (T, 96, 96) float16
            np.save(str(npy_path), roi)
            stats["success"] += 1
        except Exception as e:
            print(f"\nFAILED: {video_path} — {e}")
            stats["failed"] += 1
            # Save an empty placeholder so we know this video needs handling
            np.save(str(npy_path), np.zeros((1, TARGET_SIZE, TARGET_SIZE), dtype=np.float16))

    extractor.close()

    print(f"[{split.upper()}] Done: {stats['success']} ok, "
          f"{stats['failed']} failed, {stats['skipped']} skipped")

    # Save stats
    with open(out_path / "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default="data/",      help="Root with train/ and test/")
    parser.add_argument("--output_root", default="data/rois/", help="Output root for .npy files")
    parser.add_argument("--splits",      nargs="+", default=["train", "test"])
    parser.add_argument("--no_skip",     action="store_true",   help="Reprocess existing files")
    args = parser.parse_args()

    for split in args.splits:
        preprocess_split(
            data_root=args.data_root,
            output_root=args.output_root,
            split=split,
            skip_existing=not args.no_skip,
        )

    print("\nAll splits done. You can now run training.")
```

### 5.3 Expected Runtime

On a single CPU core, MediaPipe processes roughly 80–120 frames/sec. At 25 fps video, that's about 3–5× real-time. For a dataset of 5,000 clips averaging 3 seconds each (75 frames):
- Total frames: ~375,000
- Estimated time: ~50–75 minutes single-core
- Speed up by setting `--splits train test` and running in parallel if needed

---

## 6. Model Architecture (Full Code)

### 6.1 The Visual Frontend (`src/model.py`)

```python
"""
src/model.py

Full VSR model:
  - 3D-ResNet-18 visual frontend
  - E-Branchformer temporal encoder (12 or 18 layers)
  - CTC head + Transformer decoder
  - Joint CTC/Attention loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Visual Frontend: 3D-ResNet-18
# ═══════════════════════════════════════════════════════════════════════════════

class BasicBlock2D(nn.Module):
    """Standard ResNet-18 residual block (2D). Used after the 3D frontend."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet3DFrontend(nn.Module):
    """
    3D convolutional frontend followed by ResNet-18.

    Input:  [B, T, 1, 96, 96]
    Output: [B, T, 512]

    The 3D conv (kernel depth=5) learns to detect lip motion across 5 adjacent
    frames simultaneously — that's what makes this different from a 2D ResNet.
    After the 3D conv, the time dimension is preserved and 2D ResNet processes
    each frame independently (but with motion-aware features from the 3D stage).
    """

    def __init__(self):
        super().__init__()

        # 3D temporal front-end: captures short-range motion (5-frame window)
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=(5, 7, 7),     # (T, H, W) — temporal kernel of 5 frames
            stride=(1, 2, 2),          # Downsample spatial by 2× each dim
            padding=(2, 3, 3),         # 'same' padding in temporal, halve spatial
            bias=False,
        )
        self.bn3d   = nn.BatchNorm3d(64)
        self.relu   = nn.ReLU(inplace=True)
        self.pool3d = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        # After 3D stages: spatial 96→24 (two 2× downsamples)
        # Temporal T is preserved (stride=1 in time dimension)

        # ResNet-18 body (2D, applied per-frame)
        self.layer1 = self._make_layer(64,  64,  num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # Collapse spatial to 1×1

        self._init_weights()

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int, stride: int):
        layers = [BasicBlock2D(in_planes, planes, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock2D(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 1, 96, 96]
        returns: [B, T, 512]
        """
        B, T, C, H, W = x.shape

        # 3D stage: [B, 1, T, H, W] → [B, 64, T, H', W']
        x = x.permute(0, 2, 1, 3, 4)         # [B, 1, T, 96, 96]
        x = self.pool3d(self.relu(self.bn3d(self.conv3d(x))))
        # x: [B, 64, T, 24, 24]

        # Reshape for 2D ResNet: treat each time step as a separate image
        x = x.permute(0, 2, 1, 3, 4)         # [B, T, 64, 24, 24]
        x = rearrange(x, "b t c h w -> (b t) c h w")

        # 2D ResNet stages
        x = self.layer1(x)    # (B*T, 64, 24, 24)
        x = self.layer2(x)    # (B*T, 128, 12, 12)
        x = self.layer3(x)    # (B*T, 256, 6, 6)
        x = self.layer4(x)    # (B*T, 512, 3, 3)
        x = self.avgpool(x)   # (B*T, 512, 1, 1)
        x = x.flatten(1)      # (B*T, 512)

        # Restore time dimension
        x = rearrange(x, "(b t) d -> b t d", b=B, t=T)
        return x  # [B, T, 512]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — E-Branchformer Temporal Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class ConvolutionalGatedMLP(nn.Module):
    """
    The local branch of E-Branchformer.
    
    Uses a gated mechanism (GLU-style) + depthwise convolution to capture
    local temporal patterns. The gating lets the network learn which local
    features to suppress or amplify.
    
    cgMLP stands for: Convolutional Gated Multi-Layer Perceptron
    """

    def __init__(self, d_model: int, ff_expand: int = 4, kernel_size: int = 31):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        # Expand to 2× for gating, then pointwise project
        self.fc1   = nn.Linear(d_model, d_model * ff_expand * 2)
        # Depthwise conv: each channel independently (groups=d_model * ff_expand)
        hidden_dim = d_model * ff_expand
        self.dw    = nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                               padding=kernel_size // 2, groups=hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, d_model)
        self.drop  = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        residual = x
        x = self.norm(x)
        # Double expand, then GLU-style gating: splits into two halves
        x = self.fc1(x)                            # [B, T, 2 * D * ff_expand]
        x1, x2 = x.chunk(2, dim=-1)                # Each: [B, T, D * ff_expand]
        x1 = F.gelu(x1)
        # Depthwise conv over the time dimension
        x1 = x1.transpose(1, 2)                    # [B, D*ff_expand, T]
        x1 = self.dw(x1).transpose(1, 2)           # [B, T, D*ff_expand]
        # Gating: element-wise multiply with second branch
        x  = x1 * F.sigmoid(x2)
        x  = self.drop(self.fc2(x))
        return x + residual


class EBranchformerBlock(nn.Module):
    """
    One E-Branchformer encoder block.
    
    Architecture:
        x → LayerNorm → [Branch A: MHSA || Branch B: cgMLP] → Enhanced Merge
          → residual → FFN → residual → LayerNorm
    
    Key difference vs Conformer:
      - Branches run in PARALLEL (not sequential)
      - Enhanced merge uses a depthwise conv over the concatenated branches
        (not just linear projection or simple addition)
    
    This allows both branches to inform each other during merging, which is
    the "enhanced" part of E-Branchformer.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 ff_expand: int = 4, cgmlp_expand: int = 4,
                 conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Pre-norms (one per major sub-block)
        self.norm_attn  = nn.LayerNorm(d_model)
        self.norm_merge = nn.LayerNorm(2 * d_model)  # Before enhanced merge
        self.norm_ff    = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)

        # Branch A: Multi-head Self-Attention (global context)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        # Branch B: cgMLP (local context)
        self.cgmlp = ConvolutionalGatedMLP(d_model, ff_expand=cgmlp_expand,
                                           kernel_size=conv_kernel)

        # Enhanced merge: depthwise conv over [Branch A || Branch B] concatenation
        # This is what makes E-Branchformer "enhanced" vs basic Branchformer
        self.merge_dw = nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=3,
                                  padding=1, groups=2 * d_model)
        self.merge_fc = nn.Linear(2 * d_model, d_model)

        # Feed-forward network (Conformer-style, with half-step 0.5 scaling)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expand),
            nn.SiLU(),   # Swish: slightly better than ReLU in practice
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    [B, T, D]
        mask: [B, T] — True for PAD positions (will be ignored in attention)
        """
        # ── Parallel branches ──────────────────────────────────────────────────
        x_norm = self.norm_attn(x)

        # Branch A: self-attention
        a_out, _ = self.attn(x_norm, x_norm, x_norm,
                             key_padding_mask=mask)   # [B, T, D]
        a_out = self.attn_drop(a_out)

        # Branch B: cgMLP (note: cgmlp has its own internal norm + residual)
        b_out = self.cgmlp(x_norm)                    # [B, T, D]

        # ── Enhanced merge ─────────────────────────────────────────────────────
        # Concatenate both branch outputs: [B, T, 2D]
        merged = torch.cat([a_out, b_out], dim=-1)
        merged = self.norm_merge(merged)

        # Depthwise conv mixes the merged signal locally over time
        merged = merged.transpose(1, 2)               # [B, 2D, T]
        merged = self.merge_dw(merged).transpose(1, 2) # [B, T, 2D]

        # Project back to D and add residual
        x = x + self.merge_fc(merged)                 # [B, T, D]

        # ── Feed-forward (half-step Conformer style) ───────────────────────────
        x = x + 0.5 * self.ff(self.norm_ff(x))
        x = self.norm_final(x)
        return x


class EBranchformerEncoder(nn.Module):
    """
    Stack of N E-Branchformer blocks with sinusoidal positional encoding.
    
    N=12: ~lighter, trains faster  (~46M params in encoder)
    N=18: ~heavier, better WER     (~69M params in encoder) — recommended with 2×3090
    """

    def __init__(self, d_model: int = 256, n_layers: int = 18, n_heads: int = 4,
                 ff_expand: int = 4, conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Sinusoidal positional encoding (not learned — robust to variable T)
        self.register_buffer("pos_enc", self._sinusoidal_encoding(8192, d_model))

        self.input_norm = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EBranchformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_expand=ff_expand,
                cgmlp_expand=ff_expand,
                conv_kernel=conv_kernel,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Intermediate CTC losses at every 4th layer for InterCTC regularization
        # This is an optional but consistently helpful trick from competition winners
        # It forces intermediate layers to also produce meaningful alignments.
        self.inter_ctc_layers = set(range(3, n_layers, 4))  # {3, 7, 11, 15, ...}

    @staticmethod
    def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """Classic Vaswani et al. sinusoidal positional encoding."""
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        enc = torch.zeros(max_len, d_model)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        return enc.unsqueeze(0)  # [1, max_len, D]

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None):
        """
        x:       [B, T, D]
        lengths: [B] — actual sequence lengths (for padding mask)
        
        Returns:
            x:              [B, T, D] — final encoded features
            inter_outputs:  list of [B, T, D] tensors from intermediate layers
        """
        B, T, _ = x.shape
        x = self.input_drop(self.input_norm(x) + self.pos_enc[:, :T, :])

        # Build padding mask: True = PAD position
        mask = None
        if lengths is not None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        inter_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask)
            if i in self.inter_ctc_layers:
                inter_outputs.append(x)

        return x, inter_outputs


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Transformer Decoder (Autoregressive)
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerDecoder(nn.Module):
    """
    Standard autoregressive Transformer decoder.
    
    During training: teacher-forced (all target tokens fed simultaneously)
    During inference: greedy or beam search token by token
    
    Decoder attends to:
      1. Previous predicted tokens (causal self-attention)
      2. Encoder output (cross-attention)
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 6, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_drop = nn.Dropout(dropout)
        self.register_buffer("pos_enc",
                             EBranchformerEncoder._sinusoidal_encoding(2048, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-norm: more stable training
        )
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm   = nn.LayerNorm(d_model)
        self.out_fc = nn.Linear(d_model, vocab_size)

        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.out_fc.weight)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor | None = None,
                memory_key_padding_mask: torch.Tensor | None = None,
                tgt_key_padding_mask:    torch.Tensor | None = None
               ) -> torch.Tensor:
        """
        tgt:    [B, L] — target token indices (teacher-forced during training)
        memory: [B, T, D] — encoder output
        
        Returns: [B, L, vocab_size] — logits
        """
        L = tgt.size(1)
        x = self.pos_drop(self.embed(tgt) * math.sqrt(self.d_model) +
                         self.pos_enc[:, :L, :])

        # Causal mask: prevent attending to future tokens
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                L, device=tgt.device
            )

        x = self.layers(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        x = self.norm(x)
        return self.out_fc(x)  # [B, L, vocab_size]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Full VSR Model
# ═══════════════════════════════════════════════════════════════════════════════

VOCAB = ["<blank>", "<sos>", "<eos>"] + list("abcdefghijklmnopqrstuvwxyz") + [" "]
VOCAB_SIZE = len(VOCAB)   # 30
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
BLANK_IDX = 0
SOS_IDX   = 1
EOS_IDX   = 2


class VSRModel(nn.Module):
    """
    Full Visual Speech Recognition model.
    
    Components:
      1. ResNet3DFrontend: video → per-frame visual features [B, T, 512]
      2. Linear projection: 512 → 256
      3. EBranchformerEncoder: [B, T, 256] → [B, T, 256] with context
      4. CTC head: [B, T, 256] → [B, T, vocab_size]
      5. TransformerDecoder: [B, L, 256] → [B, L, vocab_size]
    """

    def __init__(self, d_model: int = 256, n_enc_layers: int = 18,
                 n_dec_layers: int = 6):
        super().__init__()
        self.d_model = d_model

        self.frontend  = ResNet3DFrontend()
        self.proj      = nn.Linear(512, d_model)
        self.encoder   = EBranchformerEncoder(
            d_model=d_model,
            n_layers=n_enc_layers,
        )
        self.ctc_head  = nn.Linear(d_model, VOCAB_SIZE)

        # Intermediate CTC heads (one per intermediate output layer)
        # Reuse the same projection weights for simplicity
        self.inter_ctc = nn.Linear(d_model, VOCAB_SIZE)

        self.decoder   = TransformerDecoder(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            n_layers=n_dec_layers,
        )

    def encode(self, x: torch.Tensor,
               lengths: torch.Tensor | None = None):
        """
        x: [B, T, 1, 96, 96]
        Returns: (encoder_output [B, T, D], inter_outputs list, ctc_logits [B, T, V])
        """
        feats = self.frontend(x)           # [B, T, 512]
        feats = self.proj(feats)           # [B, T, 256]
        enc_out, inter_outs = self.encoder(feats, lengths=lengths)
        ctc_logits = self.ctc_head(enc_out)  # [B, T, V]
        return enc_out, inter_outs, ctc_logits

    def forward(self, video: torch.Tensor, tgt: torch.Tensor,
                video_lengths: torch.Tensor | None = None,
                tgt_lengths:   torch.Tensor | None = None):
        """
        video:         [B, T, 1, 96, 96]
        tgt:           [B, L]   — teacher-forced target tokens
        video_lengths: [B]      — actual T per sample (for padding mask)
        tgt_lengths:   [B]      — actual L per sample
        
        Returns dict with all logits for loss computation.
        """
        B, T = video.shape[:2]

        # ── Encode ──────────────────────────────────────────────────────────────
        enc_out, inter_outs, ctc_logits = self.encode(video, lengths=video_lengths)

        # CTC: needs [T, B, V] and log probs
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)

        # Intermediate CTC log probs for InterCTC regularization
        inter_ctc_log_probs = [
            F.log_softmax(self.inter_ctc(h), dim=-1).permute(1, 0, 2)
            for h in inter_outs
        ]

        # ── Decode ──────────────────────────────────────────────────────────────
        # Build padding masks for decoder
        mem_pad_mask = None
        if video_lengths is not None:
            mem_pad_mask = (
                torch.arange(T, device=video.device).unsqueeze(0) >=
                video_lengths.unsqueeze(1)
            )

        tgt_pad_mask = None
        if tgt_lengths is not None:
            tgt_pad_mask = (
                torch.arange(tgt.size(1), device=video.device).unsqueeze(0) >=
                tgt_lengths.unsqueeze(1)
            )

        dec_logits = self.decoder(
            tgt, enc_out,
            memory_key_padding_mask=mem_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )   # [B, L, V]

        return {
            "ctc_log_probs":       ctc_log_probs,       # [T, B, V]
            "inter_ctc_log_probs": inter_ctc_log_probs, # list of [T, B, V]
            "dec_logits":          dec_logits,           # [B, L, V]
            "enc_out":             enc_out,              # [B, T, D]  (for debugging)
        }

    @torch.no_grad()
    def greedy_decode(self, video: torch.Tensor,
                      video_lengths: torch.Tensor | None = None,
                      max_decode_len: int = 200) -> list[str]:
        """
        Fast greedy decoding for validation. Not used for final submission.
        """
        self.eval()
        enc_out, _, ctc_logits = self.encode(video, lengths=video_lengths)
        ctc_preds = ctc_logits.argmax(-1)  # [B, T]

        results = []
        for b in range(ctc_preds.size(0)):
            # CTC collapse: remove blanks and consecutive duplicates
            idxs = ctc_preds[b].cpu().tolist()
            T_len = video_lengths[b].item() if video_lengths is not None else len(idxs)
            collapsed = []
            prev = -1
            for idx in idxs[:T_len]:
                if idx != BLANK_IDX and idx != prev:
                    collapsed.append(idx)
                prev = idx
            text = "".join(IDX2CHAR.get(i, "") for i in collapsed
                          if i not in (SOS_IDX, EOS_IDX))
            results.append(text.strip())
        return results

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_M": total / 1e6,
            "trainable_M": trainable / 1e6,
        }
```

### 6.2 Loss Function (`src/utils.py`)

```python
"""
src/utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointCTCAttentionLoss(nn.Module):
    """
    Joint CTC + Attention (cross-entropy) loss with InterCTC regularization.
    
    Formula:
        L = λ_ctc × L_ctc  +  (1-λ_ctc) × L_attn  +  λ_inter × Σ L_inter_ctc_i
    
    Standard values:
        λ_ctc   = 0.3   (from AutoAVSR paper)
        λ_inter = 0.3   (per intermediate layer, from InterCTC paper)
    
    InterCTC regularization forces intermediate encoder layers to produce
    good CTC alignments, acting as auxiliary supervision throughout the
    encoder depth. This prevents the "vanishing gradient in depth" problem
    where early encoder layers get almost no gradient signal.
    """

    def __init__(self, blank_idx: int = 0, pad_idx: int = -100,
                 ctc_weight: float = 0.3, inter_ctc_weight: float = 0.1):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank_idx,
            reduction="mean",
            zero_infinity=True,  # Ignore -inf CTC losses (label > input length)
        )
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=0.1,  # Label smoothing: prevents overconfidence
        )
        self.ctc_weight   = ctc_weight
        self.inter_weight = inter_ctc_weight

    def forward(self, model_output: dict,
                targets: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> dict:
        """
        model_output: dict from VSRModel.forward()
        targets:      [B, L] — ground-truth token indices
        input_lengths:  [B] — actual video frame lengths
        target_lengths: [B] — actual label lengths
        
        Returns dict with individual and total losses.
        """
        device = targets.device

        # ── CTC loss ────────────────────────────────────────────────────────────
        # CTC input lengths: the encoder doesn't temporally downsample,
        # so input_lengths pass through unchanged.
        ctc_input_lengths = input_lengths.cpu()

        # CTC targets: concatenated without padding
        ctc_targets = torch.cat([
            targets[b, :target_lengths[b]] for b in range(targets.size(0))
        ])  # [sum(L_i)]

        l_ctc = self.ctc_loss(
            model_output["ctc_log_probs"],   # [T, B, V]
            ctc_targets.cpu(),
            ctc_input_lengths,
            target_lengths.cpu(),
        )

        # ── Attention (CE) loss ─────────────────────────────────────────────────
        # Teacher forcing: predict token i from tokens 0..i-1
        # tgt input to decoder is tokens[:-1], target for loss is tokens[1:]
        dec_logits = model_output["dec_logits"]  # [B, L, V]
        # Shift: decoder input had <sos> prepended, so dec_logits[i] predicts targets[i]
        # Here we assume targets already includes <eos>, so shift by 1:
        l_attn = self.ce_loss(
            dec_logits[:, :-1, :].reshape(-1, dec_logits.size(-1)),
            targets[:, 1:].reshape(-1),
        )

        # ── InterCTC losses ─────────────────────────────────────────────────────
        l_inter_total = torch.tensor(0.0, device=device)
        for inter_log_probs in model_output.get("inter_ctc_log_probs", []):
            l_i = self.ctc_loss(
                inter_log_probs,
                ctc_targets.cpu(),
                ctc_input_lengths,
                target_lengths.cpu(),
            )
            l_inter_total = l_inter_total + l_i

        if len(model_output.get("inter_ctc_log_probs", [])) > 0:
            l_inter_total = l_inter_total / len(model_output["inter_ctc_log_probs"])

        # ── Total loss ──────────────────────────────────────────────────────────
        l_total = (self.ctc_weight * l_ctc +
                   (1 - self.ctc_weight) * l_attn +
                   self.inter_weight * l_inter_total)

        return {
            "total":  l_total,
            "ctc":    l_ctc,
            "attn":   l_attn,
            "inter":  l_inter_total,
        }
```

---

## 7. Training — DDP on 2×3090

### 7.1 Dataset (`src/dataset.py`)

```python
"""
src/dataset.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

from src.model import VOCAB, CHAR2IDX, SOS_IDX, EOS_IDX, BLANK_IDX

MAX_T = 400  # Max frames to allow (safety: clips longer than 16s would OOM)


class LipReadingDataset(Dataset):
    """
    Loads pre-extracted ROI .npy files + text transcripts.
    
    Data augmentation (training only):
      - Horizontal flip (50% chance) — lips are symmetric
      - Time masking: randomly zero out contiguous temporal windows (SpecAugment-style)
      - Spatial cutout: randomly zero out patches in the 96×96 frame
      - Speed perturbation: resample temporal axis ±10%
    """

    def __init__(self, roi_root: str, txt_root: str, split: str,
                 augment: bool = False, max_t: int = MAX_T):
        self.augment = augment
        self.max_t   = max_t
        self.samples = []

        roi_path = Path(roi_root) / split
        txt_path = Path(txt_root) / split

        for npy_file in sorted(roi_path.rglob("*.npy")):
            rel = npy_file.relative_to(roi_path)
            txt_file = txt_path / rel.with_suffix(".txt")

            if not txt_file.exists():
                # Test set: transcript not available
                self.samples.append((str(npy_file), None))
            else:
                transcript = txt_file.read_text().strip().lower()
                if len(transcript) == 0:
                    continue
                self.samples.append((str(npy_file), transcript))

        print(f"[{split}] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _text_to_tokens(self, text: str) -> torch.Tensor:
        """
        Convert transcript to token indices.
        Includes <sos> at start and <eos> at end for decoder training.
        """
        tokens = [SOS_IDX]
        for c in text:
            if c in CHAR2IDX:
                tokens.append(CHAR2IDX[c])
        tokens.append(EOS_IDX)
        return torch.tensor(tokens, dtype=torch.long)

    def _augment(self, frames: np.ndarray) -> np.ndarray:
        """
        frames: (T, 96, 96) float16
        """
        # 1. Horizontal flip (lips are left-right symmetric for VSR purposes)
        if random.random() < 0.5:
            frames = frames[:, :, ::-1].copy()

        T = frames.shape[0]

        # 2. Temporal masking (SpecAugment for video):
        #    Randomly zero out a contiguous window of frames
        if T > 20 and random.random() < 0.8:
            mask_width = random.randint(1, min(20, T // 5))
            mask_start = random.randint(0, T - mask_width)
            frames[mask_start:mask_start + mask_width] = 0.0

        # 3. Spatial cutout: zero a random rectangular region in each frame
        if random.random() < 0.5:
            h, w = 96, 96
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            dw = random.randint(10, 30)
            dh = random.randint(10, 30)
            x1, x2 = max(0, cx - dw//2), min(w, cx + dw//2)
            y1, y2 = max(0, cy - dh//2), min(h, cy + dh//2)
            frames[:, y1:y2, x1:x2] = 0.0

        # 4. Speed perturbation: resample temporal axis ±10%
        if T > 10 and random.random() < 0.5:
            rate = random.uniform(0.9, 1.1)
            new_T = max(5, int(T * rate))
            indices = np.linspace(0, T - 1, new_T).astype(int)
            frames = frames[indices]

        return frames

    def __getitem__(self, idx):
        npy_path, transcript = self.samples[idx]

        # Load pre-extracted ROI
        frames = np.load(npy_path).astype(np.float32)  # (T, 96, 96)

        # Truncate to max_t
        if frames.shape[0] > self.max_t:
            frames = frames[:self.max_t]

        # Augment (training only)
        if self.augment:
            frames = self._augment(frames)

        # Add channel dim: (T, 1, 96, 96)
        frames_t = torch.from_numpy(frames).unsqueeze(1)

        if transcript is None:
            return frames_t, None, None  # Test set

        tokens = self._text_to_tokens(transcript)
        return frames_t, tokens, npy_path


def collate_fn(batch):
    """
    Pad variable-length sequences.
    
    Returns:
        videos:        [B, T_max, 1, 96, 96]
        tokens:        [B, L_max]
        video_lengths: [B]
        token_lengths: [B]
        paths:         list of str
    """
    videos, tokens, paths = zip(*batch)

    # Filter out None tokens (test split mixed in)
    has_labels = all(t is not None for t in tokens)

    B       = len(videos)
    T_max   = max(v.shape[0] for v in videos)
    C, H, W = 1, 96, 96

    video_lens = torch.tensor([v.shape[0] for v in videos], dtype=torch.long)
    padded_videos = torch.zeros(B, T_max, C, H, W)
    for i, v in enumerate(videos):
        padded_videos[i, :v.shape[0]] = v

    if has_labels:
        L_max      = max(t.shape[0] for t in tokens)
        token_lens = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)
        padded_tokens = torch.full((B, L_max), fill_value=-100, dtype=torch.long)
        for i, t in enumerate(tokens):
            padded_tokens[i, :t.shape[0]] = t
        return padded_videos, padded_tokens, video_lens, token_lens, list(paths)
    else:
        return padded_videos, None, video_lens, None, list(paths)
```

### 7.2 Full Training Script (`src/train.py`)

```python
"""
src/train.py

Launch with:
    torchrun --nproc_per_node=2 --master_port=29500 src/train.py

This runs two processes, one per GPU.
Each process handles half the batch — effective batch doubles automatically.
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import json
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import VSRModel, VOCAB_SIZE
from src.dataset import LipReadingDataset, collate_fn
from src.utils import JointCTCAttentionLoss


# ─── Configuration ──────────────────────────────────────────────────────────────
CFG = {
    # Model
    "d_model":     256,
    "n_enc_layers": 18,
    "n_dec_layers": 6,

    # Data
    "roi_root":    "data/rois/",
    "txt_root":    "data/",
    "val_fraction": 0.05,   # Hold out 5% of train for validation

    # Training
    "epochs":      25,
    "batch_per_gpu": 16,    # Effective batch = 16 × 2 GPUs = 32
    "num_workers":  4,
    "grad_clip":    5.0,
    "warmup_steps": 2000,

    # Learning rates (layer-wise)
    "lr_frontend":  1e-5,
    "lr_encoder":   1e-4,
    "lr_decoder":   3e-4,

    # Loss weights
    "ctc_weight":   0.3,
    "inter_weight": 0.1,

    # Checkpointing
    "ckpt_dir":    "checkpoints/finetuned/",
    "save_every":   5,

    # Pretrained weights
    "pretrained_ckpt": "checkpoints/pretrained/model_vsr_lrs3vox_base_s3fd.pth",
    "load_pretrained": True,
}


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def get_param_groups(model: VSRModel, cfg: dict) -> list:
    """
    Layer-wise learning rates.
    Frontend gets lower LR (don't destroy pretrained visual features).
    Encoder + decoder get higher LR (need to adapt to new domain).
    """
    return [
        {"params": model.frontend.parameters(),  "lr": cfg["lr_frontend"]},
        {"params": model.proj.parameters(),       "lr": cfg["lr_encoder"]},
        {"params": model.encoder.parameters(),    "lr": cfg["lr_encoder"]},
        {"params": model.ctc_head.parameters(),   "lr": cfg["lr_decoder"]},
        {"params": model.inter_ctc.parameters(),  "lr": cfg["lr_decoder"]},
        {"params": model.decoder.parameters(),    "lr": cfg["lr_decoder"]},
    ]


def cosine_warmup_lr(optimizer, step: int, warmup_steps: int,
                     total_steps: int, min_lr_factor: float = 0.05):
    """
    Linear warmup for warmup_steps, then cosine decay.
    Multiplies each param group's base LR.
    """
    import math
    if step < warmup_steps:
        factor = step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg["lr"] = pg["initial_lr"] * factor


def train():
    local_rank = setup_ddp()
    is_main    = (local_rank == 0)
    device     = torch.device(f"cuda:{local_rank}")

    if is_main:
        Path(CFG["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
        print(f"Training with {dist.get_world_size()} GPUs")
        print(f"Effective batch size: {CFG['batch_per_gpu'] * dist.get_world_size()}")

    # ── Build model ──────────────────────────────────────────────────────────────
    model = VSRModel(
        d_model=CFG["d_model"],
        n_enc_layers=CFG["n_enc_layers"],
        n_dec_layers=CFG["n_dec_layers"],
    ).to(device)

    # Load pretrained weights (if available)
    if CFG["load_pretrained"] and Path(CFG["pretrained_ckpt"]).exists():
        if is_main:
            print(f"Loading pretrained weights from {CFG['pretrained_ckpt']}")
        state = torch.load(CFG["pretrained_ckpt"], map_location=device)
        # AutoAVSR may have different key names — load with strict=False
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main:
            print(f"  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    if is_main:
        params = model.count_parameters()
        print(f"Model: {params['total_M']:.1f}M params total, "
              f"{params['trainable_M']:.1f}M trainable")

    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── Datasets ─────────────────────────────────────────────────────────────────
    full_dataset = LipReadingDataset(
        roi_root=CFG["roi_root"],
        txt_root=CFG["txt_root"],
        split="train",
        augment=True,
    )

    # Simple train/val split
    n_val   = max(1, int(len(full_dataset) * CFG["val_fraction"]))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Disable augmentation on val subset
    val_ds.dataset.augment = False

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=dist.get_world_size(),
        rank=local_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_per_gpu"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ── Optimizer and scheduler ──────────────────────────────────────────────────
    param_groups = get_param_groups(model.module, CFG)
    for pg in param_groups:
        pg["initial_lr"] = pg["lr"]   # Store base LR for scheduler

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=1e-2,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    total_steps = CFG["epochs"] * len(train_loader)
    scaler      = GradScaler()  # Mixed precision scaler
    criterion   = JointCTCAttentionLoss(
        ctc_weight=CFG["ctc_weight"],
        inter_ctc_weight=CFG["inter_weight"],
    )

    # ── Training loop ────────────────────────────────────────────────────────────
    global_step = 0
    best_val_wer = float("inf")

    for epoch in range(CFG["epochs"]):
        train_sampler.set_epoch(epoch)  # Reshuffle differently each epoch
        model.train()
        epoch_losses = {"total": 0, "ctc": 0, "attn": 0, "inter": 0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"E{epoch:02d}",
                    disable=(not is_main))

        for batch in pbar:
            videos, tokens, vid_lens, tok_lens, _ = batch
            if tokens is None:
                continue   # Skip test-only samples that slipped in

            videos   = videos.to(device, non_blocking=True)
            tokens   = tokens.to(device, non_blocking=True)
            vid_lens = vid_lens.to(device)
            tok_lens = tok_lens.to(device)

            # Update LR (step-based cosine warmup)
            cosine_warmup_lr(optimizer, global_step,
                             CFG["warmup_steps"], total_steps)

            # ── Forward + backward with fp16 ────────────────────────────────────
            with autocast():
                out = model(videos, tokens, vid_lens, tok_lens)
                losses = criterion(out, tokens, vid_lens, tok_lens)

            scaler.scale(losses["total"]).backward()

            # Gradient clipping (unscale first so clip works on true magnitudes)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # ── Logging ─────────────────────────────────────────────────────────
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1
            global_step += 1

            if is_main and global_step % 50 == 0:
                avg = {k: v / n_batches for k, v in epoch_losses.items()}
                pbar.set_postfix({
                    "L": f"{avg['total']:.3f}",
                    "ctc": f"{avg['ctc']:.3f}",
                    "attn": f"{avg['attn']:.3f}",
                })

        # ── Validation ───────────────────────────────────────────────────────────
        if is_main:
            model.eval()
            val_wer = quick_val(model.module, val_loader, device)
            print(f"\nEpoch {epoch:02d} — train_loss: {epoch_losses['total']/n_batches:.4f} "
                  f"— val_WER: {val_wer:.4f}")

            # Save checkpoint
            if epoch % CFG["save_every"] == 0 or val_wer < best_val_wer:
                if val_wer < best_val_wer:
                    best_val_wer = val_wer

                ckpt = {
                    "epoch":          epoch,
                    "global_step":    global_step,
                    "best_val_wer":   best_val_wer,
                    "model_state":    model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg":            CFG,
                }
                ckpt_path = Path(CFG["ckpt_dir"]) / f"epoch_{epoch:02d}_wer{val_wer:.3f}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"Saved: {ckpt_path}")

    cleanup_ddp()


def quick_val(model: VSRModel, loader, device, max_batches: int = 50) -> float:
    """
    Approximate WER on validation set using greedy CTC decoding.
    This is fast (no beam search) but slightly optimistic compared to final WER.
    """
    import editdistance

    total_edits = 0
    total_words = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            videos, tokens, vid_lens, tok_lens, _ = batch
            if tokens is None:
                continue
            videos   = videos.to(device)
            vid_lens = vid_lens.to(device)

            preds = model.greedy_decode(videos, video_lengths=vid_lens)

            for b in range(len(preds)):
                # Decode reference
                ref_toks = tokens[b, 1:tok_lens[b]-1].tolist()  # Strip <sos> and <eos>
                from src.model import IDX2CHAR, SOS_IDX, EOS_IDX, BLANK_IDX
                ref = "".join(IDX2CHAR.get(i, "") for i in ref_toks
                              if i not in (SOS_IDX, EOS_IDX, BLANK_IDX))
                hyp = preds[b]

                ref_words = ref.split()
                hyp_words = hyp.split()
                total_edits += editdistance.eval(ref_words, hyp_words)
                total_words += max(1, len(ref_words))

    return total_edits / total_words if total_words > 0 else 1.0


if __name__ == "__main__":
    train()
```

### 7.3 Memory Budget Breakdown (2×3090, fp16)

| Component | VRAM per GPU |
|-----------|-------------|
| Model weights (fp16) | ~380 MB (VSR model ~190M params × 2 bytes) |
| Model weights fp32 copy (optimizer) | ~760 MB |
| Adam optimizer states (2 × fp32) | ~1.5 GB |
| Activations (batch=16, T=150, fp16) | ~5.5 GB |
| Gradient buffers | ~380 MB |
| **Total per GPU** | **~8.5 GB** |

With 24 GB available, you have ~15 GB headroom. You can safely increase `batch_per_gpu` to 24 if clips are short, or use 16 and leave headroom for gradient checkpointing on very long clips.

**Gradient checkpointing** (use if OOM on very long clips):
```python
# Add this to train.py if you hit OOM
model.module.encoder.gradient_checkpointing = True
# This recomputes activations during backward instead of storing them
# Cuts activation memory by ~60% at cost of ~35% compute overhead
```

---

## 8. Language Model Rescoring (KenLM + Beam Search)

### 8.1 The Math of Beam Search + LM

At each decoding timestep $t$, the beam maintains $K$ active hypotheses $\{h_1, \ldots, h_K\}$. Each hypothesis $h$ has a score:

$$\text{score}(h) = \underbrace{\log P_\text{ctc}(h \mid \text{video})}_{\text{acoustic model}} + \alpha \cdot \underbrace{\log P_\text{lm}(h)}_{\text{language model}} + \beta \cdot |h|_\text{words}$$

- $\alpha$ (LM weight): how much to trust the language model. Typical range: 0.4–0.8.
- $\beta$ (word insertion bonus): corrects for LM bias toward shorter sequences. Typical range: 0.2–0.5.
- $|h|_\text{words}$: word count of hypothesis $h$ (number of spaces + 1).

The key insight: $P_\text{ctc}$ captures what the lips *look like* they're saying; $P_\text{lm}$ captures what's *plausible in English*. Combined, they resolve viseme ambiguity far better than either alone.

### 8.2 Build the KenLM Language Model

```bash
# Download LibriSpeech LM corpus (standard choice for English VSR)
wget https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
gunzip librispeech-lm-norm.txt.gz

# Install KenLM tooling
sudo apt-get install -y build-essential cmake libboost-all-dev \
    libbz2-dev liblzma-dev zlib1g-dev
git clone https://github.com/kpu/kenlm
cd kenlm && mkdir build && cd build
cmake .. && make -j 4
cd ../..

# Build a 4-gram language model (4-gram is the sweet spot: 5-gram rarely helps,
# 3-gram gives worse perplexity, 4-gram is fast and well-calibrated)
./kenlm/build/bin/lmplz -o 4 \
    --discount_fallback \
    < librispeech-lm-norm.txt \
    > lm/librispeech_4gram.arpa

# Convert to binary format (10× faster at inference time)
./kenlm/build/bin/build_binary \
    trie \
    lm/librispeech_4gram.arpa \
    lm/librispeech_4gram.bin

# Estimated sizes:
# .arpa file: ~1.5 GB
# .bin file:  ~800 MB
```

**Domain-specific LM (do this if time allows)**: Build an *additional* n-gram LM from the competition's training transcripts. Interpolate with LibriSpeech LM:

```python
# Competition transcript LM (tiny but domain-specific)
# Extract all transcripts first:
import glob
texts = []
for f in glob.glob("data/train/**/*.txt", recursive=True):
    texts.append(open(f).read().strip().lower())
with open("lm/competition_train.txt", "w") as f:
    f.write("\n".join(texts))

# Build unigram + bigram from competition data (too small for 4-gram)
# ./kenlm/build/bin/lmplz -o 2 < lm/competition_train.txt > lm/competition_2gram.arpa
```

### 8.3 Beam Search Decoder (`src/predict.py`)

```python
"""
src/predict.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pyctcdecode import build_ctcdecoder

from src.model import VSRModel, VOCAB, BLANK_IDX
from src.dataset import LipReadingDataset, collate_fn
from src.preprocess import LipROIExtractor


def build_decoder(lm_path: str = "lm/librispeech_4gram.bin",
                  alpha: float = 0.6,
                  beta: float = 0.4,
                  beam_width: int = 80) -> object:
    """
    Build beam search decoder with KenLM language model.
    
    pyctcdecode handles:
      - CTC beam search (pruning, merging blank paths)
      - KenLM integration (word-level LM scoring at word boundaries)
      - Hotword boosting (optional — boost domain-specific vocabulary)
    
    Args:
        alpha:      LM weight (higher = trust LM more)
        beta:       Word insertion bonus (higher = longer outputs preferred)
        beam_width: Number of active hypotheses. 40–100 is practical.
                   Diminishing returns above ~100.
    """
    # pyctcdecode expects vocab without <blank> in index 0 — it handles blank internally
    # Build vocab list: index 0 = blank, rest = characters
    vocab_list = [v if v != "<blank>" else "" for v in VOCAB]

    decoder = build_ctcdecoder(
        labels=vocab_list,
        kenlm_model=lm_path,
        alpha=alpha,
        beta=beta,
        ctc_token_idx=BLANK_IDX,
    )
    return decoder


def decode_batch_ctc(log_probs_batch: list, decoder, beam_width: int = 80) -> list[str]:
    """
    Decode a list of log-prob arrays using the KenLM beam decoder.
    Parallelized with pyctcdecode's multiprocessing support.
    
    log_probs_batch: list of (T_i, vocab_size) numpy arrays
    """
    from multiprocessing.pool import Pool
    with Pool(processes=4) as pool:
        results = decoder.decode_batch(
            pool,
            [lp.astype(np.float32) for lp in log_probs_batch],
            beam_width=beam_width,
        )
    # Normalize: lowercase, collapse whitespace
    return [" ".join(r.lower().split()) for r in results]


def run_inference(
    checkpoint_path: str,
    sample_csv: str,
    output_csv: str,
    lm_path: str = "lm/librispeech_4gram.bin",
    alpha: float = 0.6,
    beta: float = 0.4,
    beam_width: int = 80,
    batch_size: int = 8,
    device: str = "cuda",
):
    """
    Full inference pipeline: video paths → submission CSV.
    """
    # ── Load model ───────────────────────────────────────────────────────────────
    model = VSRModel()
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}, "
          f"val_WER={ckpt.get('best_val_wer', '?'):.3f}")

    # ── Build decoder ────────────────────────────────────────────────────────────
    decoder = build_decoder(lm_path=lm_path, alpha=alpha, beta=beta,
                             beam_width=beam_width)
    print(f"Decoder: beam={beam_width}, α={alpha}, β={beta}")

    # ── Load sample CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(sample_csv)
    video_paths = df["path"].tolist()

    # ── ROI extractor (for videos not pre-processed) ─────────────────────────────
    extractor = LipROIExtractor()

    # ── Inference ────────────────────────────────────────────────────────────────
    predictions = []
    failed_paths = []

    # Process in mini-batches for efficiency
    for i in tqdm(range(0, len(video_paths), batch_size), desc="Inference"):
        batch_paths = video_paths[i:i + batch_size]
        batch_frames = []
        batch_valid  = []

        for path in batch_paths:
            # Check if pre-extracted ROI exists
            npy_path = Path("data/rois") / path.replace(".mp4", ".npy")
            try:
                if npy_path.exists():
                    frames = np.load(str(npy_path)).astype(np.float32)
                else:
                    # Fall back to live extraction
                    frames = extractor.extract(path).astype(np.float32)

                # (T, 96, 96) → (T, 1, 96, 96) tensor
                t = torch.from_numpy(frames).unsqueeze(1)
                batch_frames.append(t)
                batch_valid.append(True)
            except Exception as e:
                print(f"WARN: {path} → {e}")
                batch_frames.append(None)
                batch_valid.append(False)
                failed_paths.append(path)

        # Pad batch
        valid_frames = [f for f, v in zip(batch_frames, batch_valid) if v]
        if len(valid_frames) == 0:
            predictions.extend(["" for _ in batch_paths])
            continue

        T_max    = max(f.shape[0] for f in valid_frames)
        vid_lens = torch.tensor([f.shape[0] for f in valid_frames], dtype=torch.long)
        padded   = torch.zeros(len(valid_frames), T_max, 1, 96, 96)
        for j, f in enumerate(valid_frames):
            padded[j, :f.shape[0]] = f
        padded = padded.to(device)
        vid_lens = vid_lens.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, _, ctc_logits = model.encode(padded, lengths=vid_lens)
            log_probs_list = torch.log_softmax(ctc_logits, dim=-1)

        # Decode
        log_probs_np = [
            log_probs_list[j, :vid_lens[j].item()].cpu().numpy()
            for j in range(len(valid_frames))
        ]
        decoded = decode_batch_ctc(log_probs_np, decoder, beam_width=beam_width)

        # Re-insert failures as empty strings
        valid_idx = 0
        for v in batch_valid:
            if v:
                predictions.append(decoded[valid_idx])
                valid_idx += 1
            else:
                predictions.append("")

    extractor.close()

    # ── Save submission ───────────────────────────────────────────────────────────
    df["transcription"] = predictions
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(predictions)} predictions to {output_csv}")
    if failed_paths:
        print(f"WARNING: {len(failed_paths)} videos failed — check {failed_paths[:5]}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--sample_csv",  default="sample_submission.csv")
    parser.add_argument("--output_csv",  default="submission.csv")
    parser.add_argument("--lm_path",     default="lm/librispeech_4gram.bin")
    parser.add_argument("--alpha",       type=float, default=0.6)
    parser.add_argument("--beta",        type=float, default=0.4)
    parser.add_argument("--beam_width",  type=int,   default=80)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        sample_csv=args.sample_csv,
        output_csv=args.output_csv,
        lm_path=args.lm_path,
        alpha=args.alpha,
        beta=args.beta,
        beam_width=args.beam_width,
        batch_size=args.batch_size,
        device=args.device,
    )
```

### 8.4 Tuning α and β

Run a grid search on the validation set:

```python
# tune_lm.py — run on validation set to find best α and β
import itertools
from src.predict import build_decoder, decode_batch_ctc
# ... (precompute log_probs for val set, then grid search)

for alpha, beta in itertools.product(
    [0.3, 0.5, 0.6, 0.7, 0.8],
    [0.2, 0.3, 0.4, 0.5],
):
    decoder = build_decoder(alpha=alpha, beta=beta)
    results = decode_batch_ctc(val_log_probs, decoder)
    wer = compute_wer(results, val_refs)
    print(f"α={alpha:.1f} β={beta:.1f} → WER={wer:.4f}")
```

Typical optimal range: α ≈ 0.5–0.7, β ≈ 0.3–0.5. **Don't skip this step** — the difference between bad and good (α, β) is often 5–10 points absolute WER.

---

## 9. Inference and Submission Pipeline

### 9.1 Test-Time Augmentation (TTA)

For each test clip, run inference twice: once on the original and once on the **horizontally flipped** version. Average the CTC log-probability matrices before decoding:

```python
def tta_predict(model, frames: torch.Tensor, vid_lens: torch.Tensor):
    """
    frames: [B, T, 1, 96, 96]
    Returns averaged log probs: [B, T, V]
    """
    # Original
    _, _, ctc_logits_orig = model.encode(frames, lengths=vid_lens)
    lp_orig = torch.log_softmax(ctc_logits_orig, dim=-1)

    # Horizontally flipped: flip the W dimension
    frames_flip = frames.flip(-1)  # flip last dimension (W)
    _, _, ctc_logits_flip = model.encode(frames_flip, lengths=vid_lens)
    lp_flip = torch.log_softmax(ctc_logits_flip, dim=-1)

    # Average in probability space (not log space) for proper averaging
    avg_log_probs = torch.log(0.5 * lp_orig.exp() + 0.5 * lp_flip.exp())
    return avg_log_probs
```

TTA typically gives 1–3% relative WER reduction for free.

### 9.2 Model Ensembling

If you train two models with slightly different augmentation seeds (or one with Conformer and one with E-Branchformer), average their CTC probability distributions:

```python
def ensemble_predict(models: list, frames: torch.Tensor, vid_lens: torch.Tensor):
    """Ensemble N models by averaging their CTC probability distributions."""
    avg_probs = None
    for m in models:
        _, _, logits = m.encode(frames, lengths=vid_lens)
        probs = torch.softmax(logits, dim=-1)
        if avg_probs is None:
            avg_probs = probs
        else:
            avg_probs = avg_probs + probs
    avg_probs = avg_probs / len(models)
    return torch.log(avg_probs + 1e-8)
```

Ensembling two independently trained models typically reduces WER by 5–15% relative.

### 9.3 Post-Processing

Common VSR-specific mistakes the model makes, fixable with rules:

```python
import re

VSR_CORRECTIONS = {
    # Common viseme confusions that a language model doesn't always fix
    r"\bwether\b": "whether",
    r"\bwhere\b(?= we| they| he| she| it)": "were",  # Context-dependent
}

def postprocess(text: str) -> str:
    """Apply post-processing to VSR output."""
    text = text.lower().strip()

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Fix common contractions the model splits oddly
    text = text.replace("i m ", "i'm ").replace("don t ", "don't ")
    text = text.replace("can t ", "can't ").replace("it s ", "it's ")
    text = text.replace("that s ", "that's ").replace("he s ", "he's ")

    # Apply learned corrections
    for pattern, replacement in VSR_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text)

    return text.strip()
```

### 9.4 Final Submission Command

```bash
# Best checkpoint selection: pick lowest val_WER from checkpoints/finetuned/
ls -la checkpoints/finetuned/ | sort -k5

# Run final inference
python src/predict.py \
    --checkpoint checkpoints/finetuned/epoch_22_wer0.312.pt \
    --sample_csv sample_submission.csv \
    --output_csv submission_v1.csv \
    --lm_path lm/librispeech_4gram.bin \
    --alpha 0.6 \
    --beta 0.4 \
    --beam_width 100 \
    --batch_size 8

# Verify format before submission
python -c "
import pandas as pd
df = pd.read_csv('submission_v1.csv')
print(df.shape)
print(df.head(3))
assert list(df.columns) == ['path', 'transcription'], 'Bad columns!'
assert df['transcription'].isna().sum() == 0, 'NaN transcriptions!'
print('Format OK')
"
```

---

## 10. WER Optimization Tactics

### 10.1 Impact Ranking

Listed by expected WER impact (highest first):

| Tactic | Expected WER Reduction | Cost |
|--------|----------------------|------|
| Pretrained model (vs scratch) | 40–60 pts absolute | Zero (just download) |
| Fine-tuning on competition data | 10–25 pts absolute | Day 1–2 |
| KenLM beam search (vs greedy) | 5–15 pts absolute | Hours to set up |
| Beam width 80 vs 10 | 2–5 pts absolute | Free (slower inference) |
| LM weight α tuning | 2–5 pts absolute | 1 hr grid search |
| Model ensemble (2 models) | 3–8 pts relative | Needs 2 training runs |
| Test-time augmentation (TTA) | 1–3 pts relative | Free at inference |
| E-Branchformer vs Conformer | 1–2 pts absolute | Free if loading pretrained |
| Speed perturbation augmentation | 1–2 pts absolute | Already in code |
| Post-processing rules | 0.5–2 pts absolute | 30 min effort |

### 10.2 WER Analysis Tool

After getting a submission, analyze what type of errors dominate:

```python
# analyze_errors.py
import editdistance
import pandas as pd
from collections import Counter

def analyze_wer(pred_csv: str, ref_csv: str = None):
    """
    If you hold out a validation set, compare predictions to references.
    """
    pred_df = pd.read_csv(pred_csv)
    
    substitutions = Counter()
    deletions     = Counter()
    insertions    = Counter()
    
    # ... (compute word-level Levenshtein alignment, collect error types)
    # This tells you: are you mostly deleting words (too short predictions)?
    # Or substituting (wrong words)? Each has a different fix.
    
    print(f"Total WER: {total_edits/total_words:.4f}")
    print(f"Substitution rate: {S/N:.4f}")
    print(f"Deletion rate:     {D/N:.4f}")
    print(f"Insertion rate:    {I/N:.4f}")
    
    # If deletions >> insertions: model is under-predicting length.
    # Fix: decrease β (word insertion penalty) in beam search, or increase max_decode_len
    
    # If insertions >> deletions: model is over-generating.
    # Fix: increase β, or add a length penalty to the decoder.
```

### 10.3 VALLR Two-Stage Upgrade (if time permits)

If your WER after Day 2 is still above ~35%, consider this approach:

**Step 1**: Fine-tune your encoder to predict phonemes (44 English phonemes + blank) instead of characters. This is a smaller output space and often converges faster.

**Step 2**: Take the phoneme sequences from the dev/test set and feed them to a small LLM (e.g. GPT-2 small 124M, or if you have the hardware, LLaMA-3.2-1B) fine-tuned to reconstruct words from phoneme sequences.

The phoneme-to-text LLM is cheap to fine-tune because:
- It takes phoneme sequences as input (just strings like "DH AH K AE T S AE T")
- It outputs the word-level transcription
- You can generate training data from *any English text corpus* using a text-to-phoneme library (like g2p-en)

```python
# Generate phoneme→text training pairs from LibriSpeech transcripts
from g2p_en import G2p
g2p = G2p()

with open("librispeech-lm-norm.txt") as f:
    for line in f:
        words = line.strip().lower()
        phones = " ".join(g2p(words))
        # Training pair: phones → words
        # e.g. "DH AH K AE T S AE T" → "the cat sat"
```

---

## 11. Day-by-Day Action Plan

### Day 1 (Tonight) — Establish Ground Truth Fast

**Goal**: A working baseline submission + preprocessing finished + training started.

```bash
# Hour 1-2: Environment setup + data inspection
conda activate vsr
pip install -r requirements.txt  # (all packages above)
ls data/train/ | head -20        # Check data structure
python -c "
import glob
trains = glob.glob('data/train/**/*.mp4', recursive=True)
tests  = glob.glob('data/test/**/*.mp4',  recursive=True)
print(f'Train: {len(trains)} videos, Test: {len(tests)} videos')
"

# Hour 2-4: Run preprocessing (let this run while you set up training)
python src/preprocess.py --data_root data/ --output_root data/rois/
# This will take ~1 hour on 1 CPU core. 
# While it runs, set up the training script.

# Hour 4-5: Get baseline submission IMMEDIATELY
# Use the AutoAVSR pretrained model with greedy decoding — no fine-tuning yet.
# Even a mediocre baseline submission is better than nothing.
python src/predict.py \
    --checkpoint checkpoints/pretrained/model_vsr_lrs3vox_base_s3fd.pth \
    --beam_width 1 \   # Greedy for speed
    --output_csv submission_baseline.csv

# Hour 5-6: Start fine-tuning job (run overnight)
torchrun --nproc_per_node=2 --master_port=29500 src/train.py
# → This will run for ~8-12 hours for 25 epochs on dual 3090
```

### Day 2 — Fine-Tune + Language Model

**Goal**: Fine-tuned model + KenLM built + second improved submission.

```bash
# Morning: Check training progress
tail -f training.log
# Look for val_WER decreasing — if it's not moving after 5 epochs, reduce LR

# Morning: Build KenLM (parallel task)
./kenlm/build/bin/lmplz -o 4 < librispeech-lm-norm.txt > lm/librispeech_4gram.arpa
./kenlm/build/bin/build_binary trie lm/librispeech_4gram.arpa lm/librispeech_4gram.bin

# Afternoon: Evaluate available checkpoints
for ckpt in checkpoints/finetuned/*.pt; do
    python src/predict.py --checkpoint $ckpt --output_csv /tmp/pred_$(basename $ckpt).csv
    # Compute WER on val set
done

# Afternoon: Tune LM hyperparameters
python tune_lm.py  # Grid search α and β on val set

# Evening: Submit best checkpoint with KenLM
python src/predict.py \
    --checkpoint checkpoints/finetuned/best.pt \
    --lm_path lm/librispeech_4gram.bin \
    --alpha 0.6 --beta 0.4 --beam_width 80 \
    --output_csv submission_v2.csv

# Evening: Start second training run with different seed (for ensembling)
CUDA_VISIBLE_DEVICES=0 python src/train.py --seed 123 &
# (Single GPU, different seed = different model for ensemble)
```

### Day 3 — Polish + Final Submission

**Goal**: Best possible WER on final submission + write the report.

```bash
# Morning: Evaluate ensembled model
python -c "
from src.predict import ensemble_predict
# Load both trained models, ensemble their predictions
"

# Morning: Test-Time Augmentation
# Modify predict.py to use tta_predict() instead of model.encode()

# Afternoon: Final submission
python src/predict.py \
    --checkpoint checkpoints/finetuned/best_ensemble.pt \
    --lm_path lm/librispeech_4gram.bin \
    --alpha 0.65 --beta 0.45 --beam_width 100 \
    --output_csv submission_final.csv

# Verify submission format
python -c "
import pandas as pd
df = pd.read_csv('submission_final.csv')
assert list(df.columns) == ['path', 'transcription']
assert df.transcription.isna().sum() == 0
print(f'OK: {len(df)} rows')
print(df.head())
"

# Write report (use template in Section 13)
# Email to omnisub2026@mtuci.ru before 9:00 PM with:
#  - submission_final.csv
#  - report.pdf
#  - source code (zip the src/ directory)
#  - checkpoints (or a download link)
```

---

## 12. Debugging Guide

### 12.1 Common Failures and Fixes

**Problem**: MediaPipe detects no face on many clips.
```bash
# Diagnostic:
python -c "
from src.preprocess import LipROIExtractor
e = LipROIExtractor()
roi = e.extract('data/test/some_video/00001.mp4')
print('Shape:', roi.shape, 'Fail count:', e._fail_count)
"
# Fix: Lower detection confidence in face_mesh constructor to 0.3
# Fix: Check video is actually .mp4 and not corrupted: cv2.VideoCapture().isOpened()
```

**Problem**: Training loss is NaN after a few steps.
```python
# Likely cause: CTC loss with label_length > input_length
# Fix 1: Add zero_infinity=True to CTCLoss (already in code above)
# Fix 2: Check your temporal downsampling — if the model halves T,
#         then ctc_input_lengths must be halved too
# Fix 3: Clip gradients more aggressively: grad_clip = 1.0
```

**Problem**: val_WER not improving after 10 epochs.
```python
# Check 1: Is the learning rate too high? Try 1e-5 for all groups.
# Check 2: Is the model overfitting? Add more dropout (0.15)
# Check 3: Is the data shuffled? Ensure DistributedSampler.set_epoch(epoch)
# Check 4: Check a few val predictions manually — are they all blank?
#           If so, the CTC blank token is dominating — reduce ctc_weight to 0.1
```

**Problem**: Beam search is very slow at inference.
```bash
# pyctcdecode with beam_width=100 can be slow on long clips.
# Optimization 1: Use batch decoding with multiprocessing (already in code)
# Optimization 2: Reduce beam_width to 50 — diminishing returns above 80
# Optimization 3: Pre-filter clips: run greedy first, only beam-search on uncertain ones
```

**Problem**: Submission has empty transcriptions for some rows.
```python
# Every row must have a non-empty transcription (empty = WER penalty of 1.0)
# Fix: After predicting, fill empty strings with the single most common
#      short phrase from the training set (e.g. the most frequent 5-word phrase)
df["transcription"] = df["transcription"].replace("", "i don't know")
# Better fix: investigate WHY those videos failed (corrupt file? no face detected?)
```

### 12.2 Sanity Checks to Run Before Each Submission

```python
# Run this before every submission
import pandas as pd
import re

df = pd.read_csv("submission.csv")

# 1. Column check
assert list(df.columns) == ["path", "transcription"], "Wrong columns"

# 2. No missing rows (compare against sample_submission)
sample = pd.read_csv("sample_submission.csv")
assert len(df) == len(sample), f"Row count mismatch: {len(df)} vs {len(sample)}"

# 3. Path order preserved
assert (df["path"].values == sample["path"].values).all(), "Paths don't match"

# 4. No NaN / None
assert df["transcription"].isna().sum() == 0, "NaN transcriptions found"

# 5. No empty strings
empty = (df["transcription"].str.strip() == "").sum()
if empty > 0:
    print(f"WARNING: {empty} empty transcriptions — fix before submitting!")

# 6. Lowercase + no extra whitespace
for i, row in df.iterrows():
    t = row["transcription"]
    assert t == t.lower(), f"Row {i} not lowercase: {t}"
    assert t == " ".join(t.split()), f"Row {i} has extra whitespace: {t!r}"

print(f"All checks passed. {len(df)} rows ready.")
```

---

## 13. Competition Report Template

The competition requires a 2–4 page article. Use this as your skeleton:

---

### Title: Visual Speech Recognition with E-Branchformer and Language Model Rescoring

**Abstract** (150 words max):
> We present our solution to OmniSub2026, a visual-only speech recognition challenge requiring transcription of silent face videos. Our system fine-tunes a pretrained 3D-ResNet + E-Branchformer encoder-decoder model on the competition training set, achieving [X]% WER on the test partition. We demonstrate the importance of affine-stabilized lip ROI extraction, joint CTC/attention training with InterCTC regularization, and 4-gram KenLM language model rescoring. Our best submission uses an ensemble of two independently trained models with test-time augmentation.

**1. System Overview**

Describe the 4-stage pipeline: face detection → ROI stabilization → VSR model → LM decoding.

**2. Model Architecture**

- 3D-ResNet-18 frontend: explain why 3D convolutions capture motion (kernel depth=5)
- E-Branchformer encoder: explain parallel branches (global MHSA + local cgMLP), cite Kim et al. 2023
- Joint CTC/attention loss: explain why CTC alone is worse than joint (cite AutoAVSR paper)
- InterCTC regularization: cite the InterCTC paper, explain gradient signal at intermediate layers

**3. Training Details**

| Component | Value |
|-----------|-------|
| GPUs | 2 × RTX 3090 |
| Epochs | 25 |
| Optimizer | AdamW, β=(0.9, 0.98) |
| Learning rate | 1e-4 (encoder), 3e-4 (decoder), warmup 2k steps |
| Batch size | 32 (16 per GPU) |
| Precision | fp16 mixed |
| Augmentation | Horizontal flip, time masking, spatial cutout, speed perturb. |

**4. Inference and Decoding**

Explain beam search + KenLM. Report the tuned α and β values. Describe TTA and ensembling.

**5. Results**

| System | WER (val) | WER (test) |
|--------|-----------|-----------|
| Baseline (pretrained, greedy) | — | [X]% |
| + Fine-tuning | [X]% | [X]% |
| + KenLM beam search | [X]% | [X]% |
| + Ensemble + TTA | [X]% | [X]% |

**6. Analysis and Conclusions**

- What worked best (fine-tuning >> everything else)
- What you'd try with more time (VALLR phoneme approach, larger backbone)
- Known failure modes (heavy occlusion, extreme rotation, unusual accents)

**References**:
- Ma et al., "Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels", ICASSP 2023
- Kim et al., "E-Branchformer: Branchformer with Enhanced Merging for Speech Recognition", SLT 2023
- Thomas et al., "VALLR: Visual ASR Language Model for Lip Reading", ICCV 2025
- Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition", Interspeech 2020

---

## Quick Reference Card

```bash
# === PREPROCESSING (run once) ===
python src/preprocess.py

# === TRAINING (dual 3090) ===
torchrun --nproc_per_node=2 src/train.py

# === BUILD LM ===
./kenlm/build/bin/lmplz -o 4 < librispeech-lm-norm.txt > lm/4gram.arpa
./kenlm/build/bin/build_binary trie lm/4gram.arpa lm/4gram.bin

# === INFERENCE ===
python src/predict.py --checkpoint checkpoints/finetuned/best.pt \
    --lm_path lm/4gram.bin --alpha 0.6 --beta 0.4 --beam_width 80

# === VERIFY SUBMISSION ===
python -c "import pandas as pd; df=pd.read_csv('submission.csv'); \
    print(df.shape, df.isna().sum().sum())"

# === EMAIL ===
# To: omnisub2026@mtuci.ru
# Attach: submission_final.csv, report.pdf, src.zip, checkpoints (link)
# Deadline: March 19, 9:00 PM
```

---

*Document version: March 2026. Architecture choices reflect ICCV 2025 / ICASSP 2025 SOTA.*
