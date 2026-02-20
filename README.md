<div align="center">

# 🏛️ Real-ESRGAN — Restorasi Citra Naskah Babad Banyumas

**Implementasi Real-Enhanced Super-Resolution Generative Adversarial Network untuk merestorasi citra naskah kuno Babad Banyumas menggunakan Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange?style=flat-square)]()

<br/>

*Tugas Akhir — Program Studi S1 Teknik Informatika*  
*Universitas Telkom Direktorat Kampus Purwokerto · 2025*

</div>

---

## 📖 Deskripsi

Naskah Babad Banyumas merupakan warisan budaya bersejarah yang menyimpan nilai filologis tinggi. Namun, naskah-naskah ini mengalami **degradasi fisik kompleks** — blur tidak beraturan, noise berlapis, tinta pudar, kompresi JPEG, hingga kerusakan fisik akibat usia.

Proyek ini mengimplementasikan **Real-ESRGAN** (*Real-Enhanced Super-Resolution GAN*) untuk merestorasi kualitas citra naskah tersebut secara digital. Model dilatih menggunakan **dataset sintetis** yang dibangun otomatis melalui *high-order degradation pipeline*, mensimulasikan kondisi kerusakan nyata pada naskah kuno.

### Alur Sistem

```
Scraping Internet → Dataset Sintetis → Preprocessing → Training → Evaluasi → Inferensi
      ↓                   ↓                 ↓              ↓           ↓          ↓
  ~1200 gambar       LR-HR pairs        Split 70/15/15   Real-ESRGAN  PSNR/SSIM  Restored
  naskah kuno        otomatis                                          /LPIPS     Image ×4
```

---

## ✨ Fitur Utama

- 🔍 **Auto Dataset Collection** — Scraping otomatis gambar naskah kuno dari internet menggunakan `icrawler`
- 🧪 **Synthetic Dataset Builder** — Pipeline degradasi bertingkat (blur, noise, JPEG, resize) untuk membuat pasangan LR-HR secara otomatis
- 🧠 **Real-ESRGAN Architecture** — Generator RRDB (23 blocks) + U-Net Discriminator dengan Spectral Normalization
- ⚖️ **Multi-Loss Training** — Kombinasi L1 Loss, Perceptual Loss (VGG19), dan Relativistic GAN Loss
- 📊 **Komprehensif Evaluasi** — Metrik PSNR, SSIM, dan LPIPS dengan statistical significance testing
- 🖼️ **Tile-based Inference** — Proses gambar berukuran besar tanpa OOM error
- ⚡ **Mixed Precision** — Dukungan `torch.cuda.amp` untuk training lebih cepat

---

## 📊 Hasil Evaluasi

| Metrik | Bicubic Baseline | **Real-ESRGAN (Ours)** |
|--------|:----------------:|:----------------------:|
| PSNR ↑ | ~25 dB | **> 28 dB** |
| SSIM ↑ | ~0.70 | **> 0.80** |
| LPIPS ↓ | ~0.35 | **< 0.20** |

> *Nilai PSNR > 30 dB dianggap kualitas baik untuk citra natural (Wang et al., 2021)*

---

## 🗂️ Struktur Proyek

```
real-esrgan-babad-banyumas/
│
├── 📄 config.yaml                   # Semua hyperparameter & path terpusat
├── 📄 requirements.txt
│
├── 📁 scripts/
│   ├── 1_scrape_images.py           # Web scraping gambar naskah dari internet
│   ├── 2_build_dataset.py           # Buat dataset sintetis LR-HR pairs otomatis
│   ├── 3_preprocess.py              # Normalisasi, augmentasi, splitting dataset
│   ├── 4_train.py                   # Training loop Real-ESRGAN
│   ├── 5_evaluate.py                # Evaluasi PSNR, SSIM, LPIPS
│   └── 6_inference.py               # Inferensi pada citra naskah baru
│
├── 📁 src/
│   ├── 📁 models/
│   │   ├── generator.py             # RRDB Generator (23 RRDB blocks)
│   │   ├── discriminator.py         # U-Net Discriminator + Spectral Norm
│   │   └── losses.py                # L1 + Perceptual + Relativistic GAN Loss
│   ├── 📁 data/
│   │   ├── scraper.py               # Multi-source image scraper
│   │   ├── degradation.py           # High-order degradation pipeline
│   │   ├── dataset.py               # PyTorch Dataset class
│   │   └── augmentation.py          # Paired geometric & color augmentation
│   └── 📁 utils/
│       ├── metrics.py               # PSNR, SSIM, LPIPS
│       ├── image_utils.py           # Fungsi manipulasi citra
│       └── logger.py                # Training logger
│
├── 📁 data/
│   ├── raw/                         # Hasil scraping
│   ├── processed/
│   │   ├── HR/                      # High-Resolution (ground truth)
│   │   └── LR/                      # Low-Resolution (sintetis, input model)
│   └── splits/                      # train.txt, val.txt, test.txt
│
├── 📁 checkpoints/                  # Model weights
└── 📁 results/                      # Output evaluasi & inferensi
```

---

## ⚙️ Arsitektur Model

### Generator — RRDBNet
```
Input LR (128×128)
      │
  [Conv 3×3] → 64 features
      │
  [RRDB × 23]  ←── Residual-in-Residual Dense Block
  ┌─────────────────────────────┐
  │  DenseBlock 1               │
  │  DenseBlock 2   + residual  │ × 23
  │  DenseBlock 3               │
  └─────────────────────────────┘
      │
  [Conv 3×3]
      │
  [PixelShuffle ×2] → Upsample 2×
  [PixelShuffle ×2] → Upsample 2×
      │
  [Conv → LeakyReLU → Conv]
      │
Output SR (512×512)
```

### Discriminator — U-Net + Spectral Norm
```
Input → [Encoder × 5, stride=2] → [Middle] → [Decoder × 5 + Skip Connections] → Probability Map
         3→64→128→256→512→512                    512→512→256→128→64→1
```

### Fungsi Loss
```
L_total = 1.0 × L_L1  +  1.0 × L_Perceptual  +  0.1 × L_RaGAN
                              (VGG19 features)      (Relativistic)
```

---

## 🚀 Instalasi

### Prasyarat

- Python 3.10+
- CUDA 11.8+ (opsional, untuk GPU training)
- RAM minimal 8GB, VRAM minimal 6GB (GPU)

### Setup Environment

```bash
# Clone repository
git clone https://github.com/username/real-esrgan-babad-banyumas.git
cd real-esrgan-babad-banyumas

# Buat virtual environment
conda create -n real-esrgan python=3.10 -y
conda activate real-esrgan

# Install dependencies
pip install -r requirements.txt
```

---

## 📋 Cara Penggunaan

### Step 1 — Scraping Dataset

```bash
python scripts/1_scrape_images.py --config config.yaml
```

Script ini akan mengunduh otomatis gambar naskah kuno dari Google Images dan Bing Images menggunakan keyword yang sudah dikonfigurasi. Hasil disimpan di `data/raw/` beserta manifest CSV.

```
============================================
SCRAPING SELESAI
============================================
Total gambar valid  : 847
- google_images     : 523
- bing_images       : 324
Manifest tersimpan  : data/raw/manifest.csv
============================================
```

### Step 2 — Build Dataset Sintetis

```bash
python scripts/2_build_dataset.py --config config.yaml
```

Membangun 1.200 pasangan LR-HR secara otomatis melalui high-order degradation pipeline. Setiap gambar HR dicrop lalu didegradasi secara sintetis untuk menghasilkan pasangan LR-nya.

### Step 3 — Preprocessing & Splitting

```bash
python scripts/3_preprocess.py --config config.yaml
```

Memvalidasi semua pasangan dan membagi dataset menjadi train (70%) / val (15%) / test (15%).

```
============================================
DATASET SPLIT SELESAI
============================================
Total pasangan valid  : 1184
Training set          : 829  (70%)
Validation set        : 177  (15%)
Test set              : 178  (15%)
============================================
```

### Step 4 — Training

```bash
# Training dari awal
python scripts/4_train.py --config config.yaml

# Lanjut dari checkpoint
python scripts/4_train.py --config config.yaml --resume checkpoints/epoch_0100.pth
```

Progress training akan ditampilkan per epoch:

```
Epoch 42/400 [====================] 829/829
  G_loss: 0.1823 | D_loss: 0.4571
  Val PSNR: 27.34 dB | Val SSIM: 0.7891
  ✓ Best model saved (PSNR: 27.34 dB)
```

### Step 5 — Evaluasi

```bash
python scripts/5_evaluate.py \
    --config config.yaml \
    --checkpoint checkpoints/best.pth
```

Menghasilkan tabel metrik lengkap + visual comparison di `results/visuals/`.

### Step 6 — Inferensi

```bash
# Single gambar
python scripts/6_inference.py \
    --input data/naskah_baru.jpg \
    --output results/restored/ \
    --checkpoint checkpoints/best.pth

# Batch (seluruh folder)
python scripts/6_inference.py \
    --input data/naskah/ \
    --output results/restored/ \
    --checkpoint checkpoints/best.pth \
    --compare
```

```
[1/3] naskah_001.jpg
    Input  : 256 × 384 px
    Output : 1024 × 1536 px  (4×)
    Waktu  : 0.83 detik
    Saved  : results/restored/naskah_001_restored.png
```

---

## ⚙️ Konfigurasi

Semua parameter dapat diubah melalui `config.yaml` tanpa menyentuh kode:

```yaml
# Ukuran patch training
dataset:
  hr_size: 512      # Ukuran HR patch
  lr_scale: 4       # Faktor downscale (LR = 128×128)
  total_target: 1200

# Arsitektur model
model:
  generator:
    num_rrdb: 23          # Jumlah RRDB blocks
    num_features: 64

# Hyperparameter training
training:
  epochs: 400
  batch_size: 8
  optimizer:
    lr_g: 1.0e-4
    lr_d: 1.0e-4
  loss_weights:
    l1: 1.0
    perceptual: 1.0
    gan: 0.1              # Weight GAN loss lebih kecil untuk stabilitas
```

---

## 📐 Degradasi Sintetis

Pipeline degradasi bertingkat yang mensimulasikan kondisi nyata naskah kuno:

```
HR Image (bersih)
     │
     ▼ ── First-Order Degradation ──────────────────────────────────
     │   Blur: Gaussian σ∈[0.1,3.0] | Motion | Anisotropic (prob 0.8)
     │   Resize: bicubic/bilinear/area/nearest, scale∈[0.15,1.0]
     │   Noise: Gaussian std∈[0,25] | Poisson scale∈[0.05,3.0] (prob 0.7)
     │   JPEG: quality∈[30,95] (prob 0.6)
     │
     ▼ ── Second-Order Degradation (enabled) ────────────────────────
     │   Degradasi tambahan dengan parameter lebih ringan
     │
     ▼ ── Sinusoidal Variation (prob 0.2) ───────────────────────────
     │   Simulasi pola moire / banding artifacts
     │
     ▼ ── Final Bicubic Downscale ×4 ────────────────────────────────
     │
LR Image (terdegradasi, 128×128)
```

---

## 📦 Dependencies

| Package | Versi | Fungsi |
|---------|-------|--------|
| `torch` | ≥ 2.0 | Deep learning framework |
| `torchvision` | ≥ 0.15 | VGG19 untuk perceptual loss |
| `opencv-python` | ≥ 4.7 | Image processing |
| `lpips` | ≥ 0.1.4 | Perceptual similarity metric |
| `icrawler` | ≥ 0.6.6 | Web image scraping |
| `scikit-image` | ≥ 0.20 | SSIM calculation |
| `albumentations` | ≥ 1.3 | Data augmentation |

---

## 📚 Referensi

```bibtex
@inproceedings{wang2021realesrgan,
  title={Real-ESRGAN: Training Real-World Blind Super-Resolution
         with Pure Synthetic Data},
  author={Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle={ICCV Workshop},
  year={2021}
}

@inproceedings{wang2018esrgan,
  title={ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and others},
  booktitle={ECCVW},
  year={2018}
}

@inproceedings{zhang2018lpips,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and others},
  booktitle={CVPR},
  year={2018}
}
```

---

## 👤 Penulis

**Suharyadi** · NIM 2211102124  
Program Studi S1 Teknik Informatika  
Universitas Telkom Direktorat Kampus Purwokerto · 2025

---

## 📄 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE).  
Dataset naskah bersumber dari [Khastara — Perpustakaan Nasional Indonesia](https://khastara.perpusnas.go.id).

---

<div align="center">

⭐ *Jika proyek ini bermanfaat, silakan beri star di GitHub* ⭐

</div>
