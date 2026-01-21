# BBB Permeability Streamlit GUI

A Streamlit web application for predicting Blood-Brain Barrier (BBB) permeability using a sparse-label multi-task ensemble model. The interface allows users to upload ligand structures in various formats and receive BBB permeability predictions with molecular descriptor calculations.

Link: https://kapaww3te2rfsgogckbbpy.streamlit.app/

## Features

- **Single-ligand prediction interface** - Upload one ligand at a time for focused analysis
- **Multiple file format support** - Supports SDF, MOL, PDB, PDBQT, MOL2, and CSV formats
- **Automatic SMILES extraction** - Extracts SMILES strings from molecular structure files
- **RDKit descriptor computation** - Calculates all available RDKit molecular descriptors
- **BBB permeability prediction** - Two-stage multi-task ensemble model predictions
- **Mechanism probability analysis** - Provides efflux, influx, PAMPA, and CNS mechanism probabilities
- **Clean, modern interface** - Red-themed aesthetic design with simplified displays
- **Model documentation** - Comprehensive documentation page with model details

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements.txt`)

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`. Use the sidebar to navigate between pages.

### Model Artifacts

The application requires trained model artifacts to make predictions. See `ARTIFACTS_GUIDE.md` for instructions on:
- Downloading pre-trained artifacts
- Training models yourself
- Setting up artifacts for Streamlit Cloud deployment

## Project Layout

```
.
├── streamlit_app.py              # Main consolidated application file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── ARTIFACTS_GUIDE.md            # Guide for obtaining model artifacts
├── generate_artifacts.py         # Script to train models and generate artifacts
├── download_from_gdrive.py       # Utility to download artifacts from Google Drive
├── bbb_model/                    # Model package (features, utils)
│   ├── __init__.py
│   ├── features.py               # RDKit descriptor computation
│   └── utils.py                  # File I/O utilities
├── bbb_model_train.py            # Model training script
├── bbb_model_predictor.py        # Prediction class (legacy, now in streamlit_app.py)
└── artifacts/                    # Model artifacts directory (not in repo)
    ├── descriptor_cols.json
    ├── stage2_feature_cols.json
    └── models/
        ├── stage1_efflux.joblib
        ├── stage1_influx.joblib
        ├── stage1_pampa.joblib
        ├── stage1_cns.joblib
        └── stage2_bbb.joblib
```

## Usage

1. **Navigate to Ligand Prediction page** using the sidebar
2. **Upload a ligand file** in one of the supported formats:
   - SDF (Structure-Data File)
   - MOL (MDL Molfile)
   - PDB (Protein Data Bank format)
   - PDBQT (AutoDock format)
   - MOL2 (Tripos MOL2 format)
   - CSV (with 'smiles' column - only first ligand processed)
3. **View extracted SMILES** and molecular structure (if available)
4. **Compute descriptors** - Click the button to calculate RDKit descriptors
5. **View predictions** - If model artifacts are available, see BBB permeability predictions

## Model Details

The application uses a sparse-label multi-task ensemble model described in the BBB manuscript:

- **Stage 1**: Predicts auxiliary ADME tasks (efflux, influx, PAMPA, CNS)
- **Stage 2**: Uses Stage 1 predictions as features for final BBB permeability prediction
- **Architecture**: LightGBM classifiers with hyperparameter optimization
- **Training**: Cross-validated with stratified bootstrap for uncertainty quantification

## Deployment

### Streamlit Cloud

1. Push your code to a GitHub repository
2. Include `streamlit_app.py` and `requirements.txt` in the root directory
3. Add model artifacts to the `artifacts/` directory (or use file upload in Streamlit Cloud)
4. Connect repository to Streamlit Cloud
5. Deploy!

The application is a single-file deployment - all code is consolidated in `streamlit_app.py`.

## Roadmap

- **Completed:**
  - Single-ligand prediction interface
  - Multiple file format support
  - RDKit descriptor computation
  - BBB permeability predictions
  - Clean, modern UI design

- **Planned:**
  - Batch processing capabilities
  - Calibration overlays for predictions
  - Applicability domain visualizations
  - Enhanced molecular structure visualization
  - PDF/CSV export with detailed reports

## Contact

Questions, bug reports, or collaboration requests: **Dr. Sivanesan Dakshanamurthy** - sd233@georgetown.edu

---




