# Automated Coronary Artery Calcium (CAC) Scoring

A complete **medical AI pipeline** for automated coronary artery calcium scoring from cardiac CT scans. Built with **PyTorch** and **MONAI**, this project demonstrates segmentation, anatomical measurement, and clinical risk stratification — directly aligned with cardiovascular CT imaging research.

## What This Project Does

1. **Loads DICOM cardiac CT scans** and converts pixel values to Hounsfield Units (HU)
2. **Segments coronary calcium deposits** using a 3-D MONAI UNet
3. **Calculates the Agatston score** — the clinical gold standard for CAC quantification
4. **Assigns cardiovascular risk categories** (none → severe)
5. **Generates side-by-side visualizations** of original CT vs. model predictions

## Agatston Score (Clinical Standard)

The Agatston score quantifies coronary calcification:

- **Detection threshold:** pixels ≥ 130 HU
- **Minimum lesion size:** ≥ 1 mm² (reduces noise)
- **Per-lesion score:** `area (mm²) × density weight`
- **Density weights:** 130–199 HU → 1, 200–299 → 2, 300–399 → 3, ≥400 → 4

| Score | Risk Category |
|-------|---------------|
| 0 | None |
| 1–10 | Minimal |
| 11–100 | Mild |
| 101–400 | Moderate |
| >400 | Severe |

## Quick Start

### 1. Install dependencies

```bash
cd C:\CT_SCAN
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run the end-to-end demo (no external data needed)

```bash
python scripts/demo.py --num-cases 12 --epochs 8
```

This will:
- Generate 12 synthetic cardiac CT volumes with embedded calcium
- Train a MONAI UNet for 8 epochs
- Score each case and save comparison images to `outputs/predictions/`

### 3. Run unit tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
CT_SCAN/
├── config.yaml                 # All hyperparameters and paths
├── requirements.txt
├── src/cac_scoring/
│   ├── dicom_io.py             # DICOM → HU volume loading
│   ├── preprocessing.py        # Normalization, resampling, HU thresholding
│   ├── agatston.py             # Agatston score calculation engine
│   ├── model.py                # MONAI 3-D UNet builder
│   ├── dataset.py              # PyTorch dataset for training
│   ├── synthetic.py            # Synthetic CT generator (demo/training)
│   ├── inference.py            # End-to-end scoring pipeline
│   └── visualize.py            # Side-by-side CT + overlay figures
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── train.py
│   ├── predict.py
│   └── demo.py
├── tests/
│   └── test_agatston.py
└── outputs/
    ├── checkpoints/            # Saved model weights
    └── predictions/            # Scores, JSON reports, PNG visualizations
```

## Usage with Real Data

### DICOM input

Place non-contrast cardiac CT DICOM files in a folder, then:

```bash
python scripts/predict.py --input path/to/dicom_folder/
```

### HU-threshold baseline (no ML)

```bash
python scripts/predict.py --input path/to/dicom_folder/ --baseline
```

### Full training on your data

1. Prepare volumes as `.npy` files with `{hu, spacing, patient_id}` and matching `_mask.npy` labels
2. Create a `manifest.csv` (see `data/synthetic/manifest.csv` after running demo)
3. Update `config.yaml` paths
4. Train:

```bash
python scripts/train.py --epochs 50 --batch-size 2
```

## Using the orCaScore Dataset

The [orCaScore MICCAI challenge](https://orcascore.grand-challenge.org/) provides 72 real cardiac CT exams (32 train / 40 test) from four scanner vendors with expert calcium annotations.

1. Register at https://orcascore.grand-challenge.org/
2. Download non-contrast CT (CSCT) volumes and reference masks
3. Convert to the project's `.npy` + manifest format
4. Labels: 1 = LAD, 2 = LCX, 3 = RCA (binary mask: `mask > 0`)

## Key Technical Highlights (for interviews)

| Topic | Implementation |
|-------|----------------|
| **Medical imaging format** | pydicom DICOM loading with RescaleSlope/Intercept → HU |
| **Deep learning framework** | MONAI UNet (3-D), Dice+CE loss |
| **Segmentation** | Binary calcium vs. background; handles tiny high-HU deposits |
| **Clinical measurement** | Full Agatston formula with per-lesion area × density weight |
| **Risk stratification** | MESA-based CAC risk categories |
| **Visualization** | Original CT + prediction overlay + ground truth comparison |

## Sample Output

After running the demo, check `outputs/predictions/` for:

- `{case_id}_comparison.png` — side-by-side CT / prediction / ground truth
- `{case_id}_report.json` — Agatston score, lesion details, risk category
- `demo_summary.json` — all cases with ML vs. baseline vs. ground truth scores

## References

- Agatston AS, et al. *Quantification of coronary artery calcium using ultrafast computed tomography.* J Am Coll Cardiol. 1990.
- Wolterink JM, et al. *An evaluation of automatic coronary artery calcium scoring methods with cardiac CT using the orCaScore framework.* Med Phys. 2016.
- [MONAI Documentation](https://docs.monai.io/)

## License

For research and portfolio use. orCaScore dataset subject to Grand Challenge terms.
