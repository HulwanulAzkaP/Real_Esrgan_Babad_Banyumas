#!/bin/bash

# ============================================================
# Real-ESRGAN Babad Banyumas — Full Pipeline Runner
# ============================================================
# Usage:
#   bash run_pipeline.sh             # run all steps
#   bash run_pipeline.sh --skip-scrape  # skip scraping (use existing raw data)
#   bash run_pipeline.sh --only-train   # only run training + eval
# ============================================================

set -e  # exit immediately on error

# ── Colors for output ──────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── Flags ──────────────────────────────────────────────────
SKIP_SCRAPE=false
SKIP_DATASET=false
SKIP_PREPROCESS=false
SKIP_TRAIN=false
ONLY_TRAIN=false
ONLY_EVAL=false
ONLY_INFER=false
CONFIG="config.yaml"
CHECKPOINT=""

# ── Parse Arguments ────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --skip-scrape)     SKIP_SCRAPE=true ;;
    --skip-dataset)    SKIP_DATASET=true ;;
    --skip-preprocess) SKIP_PREPROCESS=true ;;
    --skip-train)      SKIP_TRAIN=true ;;
    --only-train)      ONLY_TRAIN=true; SKIP_SCRAPE=true; SKIP_DATASET=true; SKIP_PREPROCESS=true ;;
    --only-eval)       ONLY_EVAL=true;  SKIP_SCRAPE=true; SKIP_DATASET=true; SKIP_PREPROCESS=true; SKIP_TRAIN=true ;;
    --only-infer)      ONLY_INFER=true; SKIP_SCRAPE=true; SKIP_DATASET=true; SKIP_PREPROCESS=true; SKIP_TRAIN=true ;;
    --config=*)        CONFIG="${arg#*=}" ;;
    --checkpoint=*)    CHECKPOINT="${arg#*=}" ;;
    --help|-h)
      echo -e "${CYAN}Usage: bash run_pipeline.sh [OPTIONS]${NC}"
      echo ""
      echo "Options:"
      echo "  --skip-scrape       Skip image scraping step"
      echo "  --skip-dataset      Skip dataset building step"
      echo "  --skip-preprocess   Skip train/val/test split step"
      echo "  --skip-train        Skip training step"
      echo "  --only-train        Only run training (assumes data is ready)"
      echo "  --only-eval         Only run evaluation"
      echo "  --only-infer        Only run inference"
      echo "  --config=PATH       Path to config file (default: config.yaml)"
      echo "  --checkpoint=PATH   Path to checkpoint for eval/infer"
      exit 0
      ;;
  esac
done

# ── Helper Functions ───────────────────────────────────────
print_header() {
  echo ""
  echo -e "${BLUE}============================================================${NC}"
  echo -e "${BLUE}  $1${NC}"
  echo -e "${BLUE}============================================================${NC}"
}

print_step() {
  echo -e "${CYAN}[STEP]${NC} $1"
}

print_success() {
  echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
  if ! command -v "$1" &> /dev/null; then
    print_error "Command '$1' not found. Please install it first."
    exit 1
  fi
}

# ── Banner ─────────────────────────────────────────────────
echo -e "${GREEN}"
echo "  ____  _____  _    _      _____ _____ _____   _____          _   _"
echo " |  _ \|  ___|| |  | |    | ____/ ____| ____|  / ____|   /\   | \ | |"
echo " | |_) | |__  | |  | |    | |__ | (___ | |__  | |  __   /  \  |  \| |"
echo " |  _ <|  __| | |  | |    |  __| \___ \|  __| | | |_ | / /\ \ | . \` |"
echo " | |_) | |___ | |__| |    | |___  ___) | |    | |__| |/ ____ \| |\  |"
echo " |____/|_____| \____/     |_____|______|_|     \_____/_/    \_\_| \_|"
echo ""
echo "  Babad Banyumas Manuscript Restoration System"
echo -e "${NC}"

START_TIME=$(date +%s)

# ── Step 0: Check prerequisites ────────────────────────────
print_header "STEP 0 — Checking Prerequisites"

check_command python3
check_command pip

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_step "Python version: $PYTHON_VERSION"

if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
  print_success "Python version OK (>= 3.10)"
else
  print_error "Python 3.10+ required. Found: $PYTHON_VERSION"
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  print_error "Config file not found: $CONFIG"
  exit 1
fi
print_success "Config file found: $CONFIG"

# ── Step 1: Install dependencies ───────────────────────────
print_header "STEP 1 — Installing Dependencies"

print_step "Installing packages from requirements.txt..."
pip install -r requirements.txt --quiet

print_step "Verifying critical imports..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torchvision; print(f'  Torchvision: {torchvision.__version__}')"
python3 -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"
python3 -c "import lpips; print(f'  LPIPS: OK')" 2>/dev/null || print_warning "LPIPS not found, installing..."
pip install lpips --quiet

# Check CUDA
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA: Available — {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('  CUDA: Not available — will use CPU (slower)')
"

print_success "All dependencies installed."

# ── Step 2: Create folder structure ───────────────────────
print_header "STEP 2 — Creating Folder Structure"

print_step "Creating directories..."

mkdir -p real-esrgan-babad-banyumas/scripts
mkdir -p real-esrgan-babad-banyumas/src/models
mkdir -p real-esrgan-babad-banyumas/src/data
mkdir -p real-esrgan-babad-banyumas/src/utils
mkdir -p real-esrgan-babad-banyumas/data/raw
mkdir -p real-esrgan-babad-banyumas/data/processed/HR
mkdir -p real-esrgan-babad-banyumas/data/processed/LR
mkdir -p real-esrgan-babad-banyumas/data/splits
mkdir -p real-esrgan-babad-banyumas/checkpoints
mkdir -p real-esrgan-babad-banyumas/results/visuals
mkdir -p real-esrgan-babad-banyumas/results/restored

# Create .gitkeep files
touch real-esrgan-babad-banyumas/data/raw/.gitkeep
touch real-esrgan-babad-banyumas/data/processed/HR/.gitkeep
touch real-esrgan-babad-banyumas/data/processed/LR/.gitkeep
touch real-esrgan-babad-banyumas/data/splits/.gitkeep
touch real-esrgan-babad-banyumas/checkpoints/.gitkeep
touch real-esrgan-babad-banyumas/results/visuals/.gitkeep

print_success "Folder structure created."

# ── Step 3: Smoke Test — verify model imports ──────────────
print_header "STEP 3 — Verifying Model Imports (Smoke Test)"

print_step "Testing all module imports..."

python3 -c "from src.models.generator import RRDBNet; print('  Generator: OK')" || {
  print_error "Generator import failed. Check src/models/generator.py"
  exit 1
}

python3 -c "from src.models.discriminator import UNetDiscriminator; print('  Discriminator: OK')" || {
  print_error "Discriminator import failed. Check src/models/discriminator.py"
  exit 1
}

python3 -c "from src.models.losses import TotalGeneratorLoss; print('  Losses: OK')" || {
  print_error "Losses import failed. Check src/models/losses.py"
  exit 1
}

python3 -c "from src.data.degradation import DegradationPipeline; print('  Degradation: OK')" || {
  print_error "Degradation import failed. Check src/data/degradation.py"
  exit 1
}

python3 -c "from src.data.dataset import BabadBanyumasDataset; print('  Dataset: OK')" || {
  print_error "Dataset import failed. Check src/data/dataset.py"
  exit 1
}

print_step "Running forward pass smoke test..."
python3 - <<'EOF'
import torch
from src.models.generator import RRDBNet
from src.models.discriminator import UNetDiscriminator

print("  Building generator (4 RRDB for speed)...")
gen = RRDBNet(num_rrdb=4)
dummy_lr = torch.randn(1, 3, 32, 32)
sr = gen(dummy_lr)
print(f"  Generator output: {list(dummy_lr.shape)} → {list(sr.shape)}")
assert sr.shape == (1, 3, 128, 128), f"Expected [1,3,128,128] got {sr.shape}"

print("  Building discriminator...")
disc = UNetDiscriminator()
pred = disc(sr)
print(f"  Discriminator output: {list(sr.shape)} → {list(pred.shape)}")

print("  All model checks passed!")
EOF

print_success "Smoke test passed."

# ── Step 4: Scraping ───────────────────────────────────────
if [ "$SKIP_SCRAPE" = false ]; then
  print_header "STEP 4 — Scraping Images"
  print_step "Starting image scraper..."
  python3 scripts/1_scrape_images.py --config "$CONFIG"
  print_success "Scraping complete."
else
  print_warning "Skipping scraping (--skip-scrape flag set)."
fi

# ── Step 5: Build Dataset ──────────────────────────────────
if [ "$SKIP_DATASET" = false ]; then
  print_header "STEP 5 — Building HR/LR Dataset Pairs"
  print_step "Generating degraded LR images from HR sources..."
  python3 scripts/2_build_dataset.py --config "$CONFIG"
  print_success "Dataset build complete."
else
  print_warning "Skipping dataset build (--skip-dataset flag set)."
fi

# ── Step 6: Preprocess / Split ─────────────────────────────
if [ "$SKIP_PREPROCESS" = false ]; then
  print_header "STEP 6 — Splitting Dataset (Train / Val / Test)"
  print_step "Creating train.txt, val.txt, test.txt splits..."
  python3 scripts/3_preprocess.py --config "$CONFIG"
  print_success "Dataset split complete."
else
  print_warning "Skipping preprocessing (--skip-preprocess flag set)."
fi

# ── Step 7: Training ───────────────────────────────────────
if [ "$SKIP_TRAIN" = false ] && [ "$ONLY_EVAL" = false ] && [ "$ONLY_INFER" = false ]; then
  print_header "STEP 7 — Training Real-ESRGAN"

  if [ -n "$CHECKPOINT" ]; then
    print_step "Resuming training from checkpoint: $CHECKPOINT"
    python3 scripts/4_train.py --config "$CONFIG" --resume "$CHECKPOINT"
  else
    print_step "Starting training from scratch..."
    python3 scripts/4_train.py --config "$CONFIG"
  fi

  print_success "Training complete. Best checkpoint saved to checkpoints/best.pth"
else
  print_warning "Skipping training."
fi

# ── Step 8: Evaluation ─────────────────────────────────────
if [ "$ONLY_INFER" = false ]; then
  print_header "STEP 8 — Evaluating on Test Set"

  EVAL_CKPT="${CHECKPOINT:-checkpoints/best.pth}"

  if [ ! -f "$EVAL_CKPT" ]; then
    print_warning "Checkpoint not found at $EVAL_CKPT — skipping evaluation."
  else
    print_step "Running evaluation with checkpoint: $EVAL_CKPT"
    python3 scripts/5_evaluate.py --config "$CONFIG" --checkpoint "$EVAL_CKPT" --split test
    print_success "Evaluation complete. Results saved to results/evaluation_results.csv"
    print_success "Visual comparisons saved to results/visuals/"
  fi
else
  print_warning "Skipping evaluation."
fi

# ── Step 9: Inference demo ─────────────────────────────────
if [ "$ONLY_EVAL" = false ]; then
  print_header "STEP 9 — Inference Demo"

  INFER_CKPT="${CHECKPOINT:-checkpoints/best.pth}"

  if [ ! -f "$INFER_CKPT" ]; then
    print_warning "Checkpoint not found at $INFER_CKPT — skipping inference demo."
  else
    # Run inference on a few test images as demo
    TEST_SPLIT="data/splits/test.txt"
    if [ -f "$TEST_SPLIT" ]; then
      SAMPLE_IMG=$(head -1 "$TEST_SPLIT" | cut -d'|' -f1)
      print_step "Running inference on sample: $SAMPLE_IMG"
      python3 scripts/6_inference.py \
        --input "$SAMPLE_IMG" \
        --output results/restored/ \
        --checkpoint "$INFER_CKPT" \
        --compare
      print_success "Inference demo complete. Output in results/restored/"
    else
      print_warning "No test split found. Skipping inference demo."
    fi
  fi
else
  print_warning "Skipping inference."
fi

# ── Final Summary ──────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

print_header "PIPELINE SELESAI"
echo -e "${GREEN}"
echo "  Total waktu    : ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "  Output files:"
echo "    Checkpoints  : checkpoints/"
echo "    Eval results : results/evaluation_results.csv"
echo "    Visuals      : results/visuals/"
echo "    Restored     : results/restored/"
echo "    Training log : results/training.log"
echo -e "${NC}"
echo -e "${BLUE}============================================================${NC}"