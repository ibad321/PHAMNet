# PHAMNet

**Multiscale Feature Learning with Parallel Hybrid Attention for Fundus Disease Recognition**

PHAMNet is a convolutional network for four-class fundus image classification
(Normal, Diabetic Retinopathy, Glaucoma, Cataract). It has three components:

1. **Multiscale Feature Fusion Stem (MFFS)**: parallel 3x3, 5x5 and 7x7 convolutions
   (32 channels each, BatchNorm + ReLU) are concatenated (96 channels) and projected
   to three channels with a 1x1 convolution.
2. **EfficientNet-B4 backbone**: ImageNet weights, classification top removed, fully
   trainable; 224x224 input gives a 7x7x1792 feature map.
3. **Hybrid Attention Module (HAM)**: a spatial attention block (SAB) and a global
   channel attention block (GCAB) act in parallel on the same feature map and are
   combined by an attention-weighted average,
   `F_out = alpha * (F * A_s) + beta * (F * A_c)`, with `alpha` and `beta` learnable
   scalars initialised to 0.5.

The head is GlobalMaxPooling, BatchNorm, Dense(256, ReLU, L2 0.016, L1 activity 0.006),
Dropout(0.45) and Dense(4, softmax).

## Repository layout

```
phamnet/
  config.py       hyper-parameters, model variants, presets (no TensorFlow import)
  data.py         dataset discovery and stratified 80/10/10 split
  duplicates.py   near-duplicate audit of the image collection
  experiment.py   data preparation shared by launcher and workers
  layers.py       SAB, GCAB, HAM, CBAM, BAM, MFFS and helper layers
  model.py        build_model(): every variant of the study
  pipeline.py     tf.data input pipeline and augmentation
  trainer.py      training and evaluation of one model / one seed
  metrics.py      accuracy, F1, MCC, AUROC, ECE, bootstrap CIs, exact McNemar test
  analysis.py     summary tables, paired tests, confusion matrices, FLOPs
  plots.py        figures
  runner.py       launcher: one worker process per model and seed
  cli.py          command-line interface
tests/            unit tests (CPU only, no dataset needed)
```

## Installation

```bash
git clone https://github.com/ibad321/PHAMNet.git
cd PHAMNet
pip install -e .          # or: pip install -r requirements.txt
```

Python 3.9+ and TensorFlow 2.16+ (Keras 3) are required. A GPU is strongly recommended.

## Data

The code expects a folder with one sub-folder per class; the sub-folder names must
contain `cataract`, `diabetic`, `glaucoma` and `normal`:

```
dataset/
  cataract/  diabetic_retinopathy/  glaucoma/  normal/
```

The study uses the 4,217-image public "Eye Diseases Classification" fundus dataset
on Kaggle. The split is a stratified 80/10/10 partition with a fixed seed
(3,375 / 420 / 422 images).

## Usage

```bash
# one model, one seed
python -m phamnet train --data-dir /path/to/dataset --runs phamnet --seeds 42

# several models and seeds, or a named preset
python -m phamnet train --data-dir /path/to/dataset --preset main
python -m phamnet train --data-dir /path/to/dataset --runs b4_tl b4_ham mffs_b4 --seeds 42 7 2024

# summary tables, McNemar tests, confusion matrices and complexity of finished runs
python -m phamnet summarize --data-dir /path/to/dataset --out-dir phamnet_results

# near-duplicate audit and split statistics only (does not import TensorFlow)
python -m phamnet audit --data-dir /path/to/dataset

# quick installation check on a few images, CPU, no pretrained weights
python -m phamnet train --data-dir /path/to/dataset --runs phamnet --seeds 1 --smoke
```

Every model and seed is trained in its own worker process. A finished run is skipped
when the command is launched again, and a worker killed for lack of memory is
restarted with a smaller batch size. Raw worker logs are written to
`phamnet_results/raw_logs/`.

On Kaggle the dataset folder is found automatically under `/kaggle/input`.

### Model variants

| name | description |
|---|---|
| `b4_tl` | EfficientNet-B4 with transfer learning (baseline) |
| `b4_sab` | + spatial attention only |
| `b4_gcab` | + channel attention only |
| `b4_ham` | + HAM, without MFFS |
| `mffs_b4` | MFFS + B4, no attention |
| `phamnet` | MFFS + B4 + HAM (full model) |
| `b4_ham_fixed` | PHAMNet with `alpha = beta = 0.5` frozen |
| `phamnet_k3` | PHAMNet with a 3x3 SAB kernel |
| `phamnet_linproj` | PHAMNet with a linear MFFS 1x1 projection |
| `b4_cbam` | + CBAM (sequential channel, then spatial) |
| `b4_bam` | + BAM (parallel branches, one summed map) |
| `vgg16_tl` | VGG-16 with transfer learning (learning rate 1e-4) |

### Training settings

Adamax (learning rate 3e-4), categorical cross-entropy, batch size 64, at most 50 epochs,
early stopping on validation loss (patience 30, best weights restored),
ReduceLROnPlateau (factor 0.5, patience 5, minimum 1e-6). Augmentation on training
images only: horizontal flip, rotation up to 15 degrees, zoom up to 10 %, brightness
shift of 0.2. The test set is used once after training. All values can be changed
through `TrainConfig` in `phamnet/config.py` or the command-line options
(`--epochs`, `--batch-size`, `--lr`).

### Outputs

Each run writes `metrics.json` (accuracy with bootstrap and exact confidence intervals,
macro precision / recall / F1, MCC, kappa, per-class sensitivity / specificity / PPV /
NPV / F1 / AUROC, ECE, Brier score, confusion matrix, learned `alpha` / `beta`, latency),
`test_probs.npy`, `history.csv` and training curves. The output folder also contains `split.csv`, `near_duplicate_pairs.csv` and
`REPORT_*.txt`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you use this code, please cite the PHAMNet paper (citation details will be added
on publication).
