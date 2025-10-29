# OpenVLA Fine-Tuning Presentation

This directory contains a Beamer presentation for the OpenVLA fine-tuning project update.

## Contents

- `presentation.tex` - Main LaTeX Beamer presentation
- `generate_figures.py` - Python script to generate all figures and charts
- `Makefile` - Build automation
- `figures/` - Generated figures (created automatically)

## Quick Start

### Generate All Figures and Compile PDF

```bash
make all
```

This will:
1. Run `generate_figures.py` to create all charts and visualizations
2. Compile the LaTeX presentation twice (for proper TOC)
3. Generate `presentation.pdf`

### Individual Steps

Generate figures only:
```bash
make figures
# or manually:
python3 generate_figures.py
```

Compile PDF only (after figures exist):
```bash
make pdf
# or manually:
pdflatex presentation.tex
pdflatex presentation.tex
```

Clean auxiliary files:
```bash
make clean
```

## Generated Figures

The `generate_figures.py` script creates:

1. **loss_comparison.png** - Bar charts comparing accuracy and L1 loss across different loss functions
2. **improvement_chart.png** - Bar charts showing improvement over base model
3. **per_dim_errors.png** - Per-dimension action prediction errors
4. **training_curves.png** - Simulated training loss and accuracy curves
5. **dataset_sample.png** - Sample image from the HDF5 dataset
6. **sample_0.png, sample_1.png, sample_2.png, sample_3.png** - Multiple trajectory samples
7. **vla_pipeline.png** - OpenVLA pipeline diagram
8. **lora_diagram.png** - LoRA fine-tuning strategy diagram
9. **openvla_architecture.png** - Overall architecture diagram

## Requirements

- Python 3.7+
- matplotlib
- numpy
- h5py
- LaTeX distribution with Beamer (e.g., TeX Live, MiKTeX)

Install Python dependencies:
```bash
pip install matplotlib numpy h5py
```

## Presentation Details

- **Duration:** ~10 minutes
- **Format:** Wide (16:9 aspect ratio)
- **Theme:** Madrid with default color scheme
- **Sections:**
  1. Introduction
  2. Dataset
  3. Methodology
  4. Experiments
  5. Results
  6. Discussion
  7. Future Work
  8. Conclusion
  9. Backup Slides

## Customization

Edit `presentation.tex` to:
- Update author name on title slide
- Adjust content in each frame
- Add/remove slides
- Change theme/colors

Edit `generate_figures.py` to:
- Update results path if needed
- Adjust plot styles
- Add new visualizations

## Viewing

After compilation, open `presentation.pdf` with any PDF viewer.

For presentation mode:
- Use a PDF viewer that supports full-screen mode (e.g., Adobe Reader, Okular, PDF.js)
- Navigate with arrow keys or page up/down

## Notes

- The presentation includes backup slides with additional technical details
- Results are pulled from: `/work/hdd/bfdj/tzhou4/checkpoints/loss_comparison_4gpu_20251029_140834/`
- Dataset samples are extracted from: `/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train/`
- If these paths change, update them in `generate_figures.py`

