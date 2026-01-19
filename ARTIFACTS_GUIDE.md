# How to Get Model Artifacts

The BBB prediction model requires trained model artifacts to make predictions. You have **two options** to obtain these artifacts:

## Option 1: Download Pre-trained Artifacts (Recommended if Available)

If the artifacts were shared on Google Drive or another location:

1. **Check the original Google Colab notebook:**
   - Original notebook: https://colab.research.google.com/drive/1rqny2wr_srdvvZiNp7GxovNUCGvQBOLD
   - The artifacts may have been saved there or linked to a Google Drive folder

2. **Use the download script:**
   ```bash
   python download_from_gdrive.py <GOOGLE_DRIVE_FOLDER_ID_OR_URL> -o artifacts
   ```

3. **Manual download:**
   - If you have a Google Drive link to the artifacts folder
   - Download all files and place them in the `artifacts/` directory with this structure:
     ```
     artifacts/
     ├── descriptor_cols.json
     ├── stage2_feature_cols.json
     └── models/
         ├── stage1_efflux.joblib
         ├── stage1_influx.joblib
         ├── stage1_pampa.joblib
         ├── stage1_cns.joblib
         └── stage2_bbb.joblib
     ```

## Option 2: Train the Models Yourself

If you have access to the training data, you can generate the artifacts by training the models.

### Requirements:

1. **Training data CSV files** with two columns:
   - `smiles`: SMILES strings of molecules
   - `label`: Binary labels (0 or 1)
   
   You need the following datasets:
   - `bbb_internal.csv` - BBB internal dataset
   - `efflux.csv` - Efflux mechanism dataset
   - `influx.csv` - Influx mechanism dataset
   - `pampa.csv` - PAMPA dataset
   - `cns.csv` - CNS dataset
   - `bbbp_external.csv` - (optional) External validation dataset

2. **Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install hyperopt  # Required for hyperparameter optimization
   ```

### Steps to Train:

1. **Prepare your training data:**
   - Create a `data/` directory (or specify your own)
   - Place all CSV files with `smiles` and `label` columns in that directory

2. **Run the training script:**
   ```bash
   python generate_artifacts.py
   ```
   
   Or run directly:
   ```bash
   python bbb_model_train.py
   ```
   
   (You'll need to edit the paths in `bbb_model_train.py` first)

3. **Wait for training to complete:**
   - Hyperparameter optimization can take several hours depending on `n_trials`
   - Model training will follow
   - Artifacts will be saved to `artifacts/` directory

### Training Parameters:

- `n_trials`: Number of hyperparameter optimization trials (default: 100, can take hours)
- `n_splits`: Cross-validation folds (default: 5)
- `seed`: Random seed for reproducibility (default: 42)

## Verify Artifacts

After obtaining artifacts, verify they exist:

```bash
ls artifacts/
# Should show: descriptor_cols.json, stage2_feature_cols.json

ls artifacts/models/
# Should show: stage1_efflux.joblib, stage1_influx.joblib, stage1_pampa.joblib, stage1_cns.joblib, stage2_bbb.joblib
```

## Using with Streamlit App

Once you have the artifacts:

1. **Local deployment:**
   - Make sure `artifacts/` directory is in the same directory as `streamlit_app.py`
   - Run: `streamlit run streamlit_app.py`

2. **Streamlit Cloud deployment:**
   - Upload the `artifacts/` directory to your GitHub repository
   - Make sure it's in the root directory (same level as `streamlit_app.py`)
   - Deploy to Streamlit Cloud - it will automatically find the artifacts

## Need Help?

- Check the original manuscript for details on the training data
- Contact: Dr. Sivanesan Dakshanamurthy — sd233@georgetown.edu
- Review the training code in `bbb_model_train.py` for more details
