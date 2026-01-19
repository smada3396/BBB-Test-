"""
BBB Permeability Prediction Streamlit App
Consolidated single-file application for Streamlit Cloud deployment.

This app provides a GUI for predicting BBB permeability using a sparse-label multi-task ensemble model.
Users can upload ligands as SMILES strings or CSV files to get BBB permeability predictions.
"""

import json
import math
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors

# Try to import Draw module - may fail on systems without X11 libraries (e.g., Streamlit Cloud)
try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except ImportError:
    # Draw module not available - visualization will be disabled
    Draw = None
    DRAW_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="BBB Permeability Studio",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "Report a bug": "https://github.com/your-org/bbb-gui/issues",
        "About": "Sparse-label multi-task learning workflow for BBB permeability modelling.",
    },
)

# ============================================================================
# BBB MODEL - FEATURES MODULE
# ============================================================================

def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Create RDKit molecule object from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def rdkit_descriptor_names() -> List[str]:
    """Get list of all RDKit descriptor names."""
    return [name for name, _fn in Descriptors._descList]


def compute_rdkit_descriptors(smiles_list: List[str]) -> pd.DataFrame:
    """
    Compute RDKit descriptors for a list of SMILES strings.
    Returns a DataFrame of RDKit descriptors.
    Rows align with smiles_list (including invalid; invalid rows become NaN then filled).
    """
    names_fns = Descriptors._descList
    names = [n for n, _ in names_fns]

    rows = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        if mol is None:
            rows.append([np.nan] * len(names))
            continue
        vals = []
        for _name, fn in names_fns:
            try:
                v = fn(mol)
                # guard against inf
                if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
                    v = np.nan
                vals.append(v)
            except Exception:
                vals.append(np.nan)
        rows.append(vals)

    df = pd.DataFrame(rows, columns=names)
    # Fill NaN with column medians for stability (tree models handle this, but keep clean)
    for c in df.columns:
        med = df[c].median()
        if np.isnan(med):
            med = 0.0
        df[c] = df[c].fillna(med)
    return df


def variance_filter(df: pd.DataFrame, min_var: float = 1e-12) -> pd.DataFrame:
    """Filter descriptors by minimum variance."""
    variances = df.var(axis=0)
    keep = variances[variances > min_var].index.tolist()
    return df[keep]


def correlation_prune(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove one feature from each pair with |corr| > threshold (keeps earlier columns)."""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=drop_cols, errors="ignore")


def make_descriptor_matrix(
    smiles: List[str],
    *,
    do_variance_filter: bool = True,
    do_corr_prune: bool = True,
    corr_threshold: float = 0.95,
) -> pd.DataFrame:
    """Create descriptor matrix from SMILES with optional filtering."""
    X = compute_rdkit_descriptors(smiles)
    if do_variance_filter:
        X = variance_filter(X)
    if do_corr_prune:
        X = correlation_prune(X, threshold=corr_threshold)
    return X

# ============================================================================
# BBB MODEL - UTILS MODULE
# ============================================================================

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    """Save object to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    """Load object from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_joblib(obj: Any, path: str) -> None:
    """Save object using joblib."""
    joblib.dump(obj, path)


def load_joblib(path: str) -> Any:
    """Load object using joblib."""
    return joblib.load(path)


def load_labeled_csv(path: str) -> pd.DataFrame:
    """Load CSV file with smiles and label columns."""
    df = pd.read_csv(path)
    if "smiles" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain columns: smiles, label")
    df = df[["smiles", "label"]].copy()
    df["label"] = df["label"].astype(int)
    return df

# ============================================================================
# BBB MODEL - PREDICTOR
# ============================================================================

@dataclass
class BBBPredictor:
    """BBB Permeability Predictor using sparse-label multi-task ensemble."""
    artifacts_dir: str = "artifacts"
    _models_loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Load model artifacts. This will raise FileNotFoundError if artifacts are missing."""
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all model artifacts."""
        if self._models_loaded:
            return
        
        self.descriptor_cols = load_json(f"{self.artifacts_dir}/descriptor_cols.json")
        self.stage2_cols = load_json(f"{self.artifacts_dir}/stage2_feature_cols.json")

        self.stage1_efflux = load_joblib(f"{self.artifacts_dir}/models/stage1_efflux.joblib")
        self.stage1_influx = load_joblib(f"{self.artifacts_dir}/models/stage1_influx.joblib")
        self.stage1_pampa = load_joblib(f"{self.artifacts_dir}/models/stage1_pampa.joblib")
        self.stage1_cns = load_joblib(f"{self.artifacts_dir}/models/stage1_cns.joblib")
        self.stage2_bbb = load_joblib(f"{self.artifacts_dir}/models/stage2_bbb.joblib")
        
        self._models_loaded = True

    def _X_desc(self, smiles: List[str]) -> pd.DataFrame:
        """Compute and align descriptor matrix."""
        canon = [canonicalize_smiles(s) for s in smiles]
        if any(c is None for c in canon):
            # keep alignment, but invalid rows become zeros
            canon = [c if c is not None else "" for c in canon]

        X = compute_rdkit_descriptors(canon)

        # align descriptor columns
        for c in self.descriptor_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[self.descriptor_cols]
        return X

    def predict_proba(self, smiles: List[str]) -> pd.DataFrame:
        """Predict BBB permeability probabilities."""
        X = self._X_desc(smiles)

        p_eff = self.stage1_efflux.predict_proba(X)[:, 1]
        p_inf = self.stage1_influx.predict_proba(X)[:, 1]
        p_pam = self.stage1_pampa.predict_proba(X)[:, 1]
        p_cns = self.stage1_cns.predict_proba(X)[:, 1]

        X2 = X.copy()
        X2["p_efflux_mech"] = p_eff
        X2["p_influx_mech"] = p_inf
        X2["p_pampa_mech"] = p_pam
        X2["p_cns_mech"] = p_cns

        # align Stage 2 columns
        for c in self.stage2_cols:
            if c not in X2.columns:
                X2[c] = 0.0
        X2 = X2[self.stage2_cols]

        p_bbb = self.stage2_bbb.predict_proba(X2)[:, 1]

        out = pd.DataFrame(
            {
                "smiles": smiles,
                "p_bbb_plus": p_bbb,
                "p_efflux_mech": p_eff,
                "p_influx_mech": p_inf,
                "p_pampa_mech": p_pam,
                "p_cns_mech": p_cns,
            }
        )
        return out

    def predict_label(self, smiles: List[str], threshold: float = 0.5) -> pd.DataFrame:
        """Predict BBB permeability labels at given threshold."""
        out = self.predict_proba(smiles)
        out["bbb_pred"] = (out["p_bbb_plus"].values >= threshold).astype(int)
        return out

# ============================================================================
# STREAMLIT PAGES
# ============================================================================

def load_predictor(artifacts_dir: str = "artifacts"):
    """Load the BBB predictor model (cached)."""
    try:
        if not os.path.exists(artifacts_dir):
            return None, f"Artifacts directory '{artifacts_dir}' not found. Please ensure model artifacts are available."
        
        predictor = BBBPredictor(artifacts_dir=artifacts_dir)
        return predictor, None
    except FileNotFoundError as e:
        return None, f"Model artifact not found: {e}. Please ensure all model files are in the artifacts directory."
    except Exception as e:
        return None, f"Error loading predictor: {str(e)}\n\n{traceback.format_exc()}"


def render_home_page():
    """Render the home/dashboard page."""
    st.title("Blood–Brain Barrier (BBB) Permeability Studio")
    st.caption(
        "Prototype interface for the sparse-label multi-task ensemble described in the BBB manuscript."
    )

    st.sidebar.header("Project Snapshot")
    st.sidebar.markdown(
        """
        - **Model focus:** Calibrated BBB permeability classification  
        - **Architecture:** Masked multi-task ensemble blended with a single-task baseline  
        - **External validation:** BBBP and out-of-source (OOS) panels  
        - **Status:** Home, documentation, and ligand prediction pages available
        """
    )
    st.sidebar.success("✓ Interactive ligand screening tab now available!")

    st.markdown(
        """
        ## Why this app exists
        Drug discovery teams struggle to predict whether small molecules cross the blood–brain barrier.
        The manuscript's fourth tab introduces a sparse-label multi-task (MT) learning workflow that blends
        auxiliary ADME tasks (PAMPA, PPB, efflux) with a calibrated single-task (ST) baseline. The blended
        predictor improves both external generalization and probability calibration, addressing two recurring
        issues in BBB screening campaigns.
        """
    )

    st.markdown(
        """
        ### Model highlights from Tab 4
        - **Sparse-label MT training:** Each auxiliary task contributes signal only where assays exist, avoiding label deletion or imputation bias.
        - **Stacked calibration:** MT logits are linearly blended with the ST baseline before post-hoc calibration selected on the development fold.
        - **Reproducibility guardrails:** All tables/figures originate from `results/metrics_clean_fixed.json`, with scripted pipelines and stratified bootstraps (B = 2000).
        """
    )

    st.divider()

    st.markdown("## Performance at a glance (Tab 4 metrics)")

    internal_metrics = {"PR-AUC": 0.915, "ROC-AUC": 0.864, "ΔPR-AUC vs ST": "+0.102"}
    external_metrics = [
        {
            "dataset": "BBBP",
            "PR-AUC": 0.950,
            "ΔPR-AUC vs ST": "+0.155",
            "p-value": "< 0.001",
        },
        {
            "dataset": "Out-of-source (OOS)",
            "PR-AUC": 0.944,
            "ΔPR-AUC vs ST": "+0.185",
            "p-value": "< 0.001",
        },
    ]

    col_internal, col_ext_1, col_ext_2 = st.columns(3)
    with col_internal:
        st.metric("Internal PR-AUC", internal_metrics["PR-AUC"], internal_metrics["ΔPR-AUC vs ST"])
        st.metric("Internal ROC-AUC", internal_metrics["ROC-AUC"])

    with col_ext_1:
        st.metric("BBBP PR-AUC", external_metrics[0]["PR-AUC"], external_metrics[0]["ΔPR-AUC vs ST"])
        st.caption(f"One-sided ΔPR-AUC p-value {external_metrics[0]['p-value']}")

    with col_ext_2:
        st.metric(
            "OOS PR-AUC",
            external_metrics[1]["PR-AUC"],
            external_metrics[1]["ΔPR-AUC vs ST"],
        )
        st.caption(f"One-sided ΔPR-AUC p-value {external_metrics[1]['p-value']}")

    st.markdown(
        """
        Calibration improves alongside discrimination: the blended model reports lower Brier score and
        expected calibration error (ECE) than the single-task baseline, with reliability diagrams approaching
        the identity line across internal and external datasets.
        """
    )

    st.divider()

    st.markdown("## From Tab 5: evaluation protocol & upcoming assets")

    tab5_col1, tab5_col2 = st.columns(2)
    with tab5_col1:
        st.subheader("Evaluation blueprint")
        st.markdown(
            """
            - **Primary metric:** Precision–recall AUC (PR-AUC); ROC-AUC reported as a secondary view.  
            - **Uncertainty:** Stratified bootstrap (B = 2000, seed = 42) yields 95% confidence intervals and ΔPR-AUC hypothesis tests.  
            - **Calibration checks:** Brier score, ECE, and reliability diagrams with equal-mass binning; Platt vs isotonic selected on the development fold.  
            - **Applicability domain:** Coverage vs precision curves using ensemble variance or representation distance thresholds.  
            """
        )

    with tab5_col2:
        st.subheader("Assets in progress")
        st.markdown(
            """
            - External & internal ROC/PR curves with confidence bands  
            - Calibration dashboards (reliability diagrams, ΔECE summaries)  
            - Confusion matrices at 0.5 and Youden thresholds  
            - Feature attribution (SHAP) views for top ADME descriptors  
            - Applicability domain plots showing precision vs coverage trade-offs  
            """
        )

    st.info(
        "🎯 **Ready to predict!** Use the 'Ligand Prediction' page in the sidebar to upload your SMILES strings or CSV files and get BBB permeability predictions."
    )

    st.markdown(
        """
        ---
        ### Roadmap
        1. **✓ Completed** – Communication spine: home and documentation pages summarizing the Tab 4–5 manuscript content.  
        2. **✓ Completed** – Ligand intake tab with SMILES/CSV upload, descriptor generation, and model scoring (see Ligand Prediction page).  
        3. **Planned** – Calibration overlay for user-submitted batches and automated report exports (PDF/CSV).
        """
    )


def render_documentation_page():
    """Render the documentation page."""
    st.title("Documentation & Runbook")
    st.caption("Reference material derived from Tabs 4–5 of the BBB manuscript.")

    st.markdown(
        """
        ## Purpose
        This application packages the manuscript's sparse-label multi-task (MT) modelling workflow into a
        Streamlit interface. The current release focuses on communication: summarising study context,
        evaluation methodology, and planned visual assets before the ligand submission module comes online.
        """
    )

    st.markdown(
        """
        ## Repository structure
        ```
        .
        ├── streamlit_app.py         # Single consolidated app file
        ├── requirements.txt         # Dependencies
        └── artifacts/               # Model artifacts (not in repo)
        ```
        """
    )

    st.markdown(
        """
        ## Local setup
        1. Create and activate a virtual environment (conda, venv, or poetry).  
        2. Install dependencies: `pip install -r requirements.txt`.  
        3. Launch the app: `streamlit run streamlit_app.py`.  
        4. Streamlit will open at `http://localhost:8501`. Use the sidebar to switch between pages.
        """
    )

    st.markdown(
        """
        ## Model overview (Tab 4 recap)
        - **Training data:** BBB permeability labels plus auxiliary ADME assays (PAMPA, PPB, efflux).  
        - **Learning strategy:** Masked MT ensemble blended with an ST baseline; losses applied only where task labels exist.  
        - **Calibration:** Platt vs isotonic assessed on the development fold; chosen calibrator reused for internal/external evaluations.  
        - **Reproducibility:** All metrics/figures regenerate from `results/metrics_clean_fixed.json`; stratified bootstrap (B = 2000) underpins confidence intervals and ΔPR-AUC tests.
        """
    )

    st.markdown(
        """
        ## Evaluation protocol (Tab 5 recap)
        - **Primary metric:** PR-AUC (robust to class imbalance); ROC-AUC reported for context.  
        - **Operational thresholds:** Summaries at 0.5 and Youden (≈0.793) include accuracy, sensitivity, specificity, F1, and MCC.  
        - **Calibration diagnostics:** Brier score, expected calibration error (ECE), reliability diagrams with equal-mass bins.  
        - **Applicability domain:** Precision vs coverage curves thresholded on ensemble variance or representation distance.  
        - **Feature interpretation:** SHAP beeswarm/waterfall plots planned for top descriptors (LightGBM head).
        """
    )

    st.markdown(
        """
        ## Roadmap
        - **✓ Completed:** Ligand intake tab supporting SMILES/CSV uploads, descriptor generation, and scoring.  
        - **Planned visual assets:** External/internal ROC & PR with CI bands, calibration dashboards, confusion matrices, SHAP explorer, AD curves.  
        - **Reporting:** Automated PDF/CSV exports once ligand scoring is active.
        """
    )

    st.info(
        "Need to contribute? Fork the GitHub repository, branch from `main`, and submit a pull request. "
        "Include before/after screenshots when adding new widgets to keep the review focused."
    )

    st.success("Questions? Open an issue via the menu or tag the modelling team on Slack.")


def render_ligand_prediction_page():
    """Render the ligand prediction page."""
    st.title("🧪 BBB Permeability Prediction")
    st.markdown(
        """
        Upload your ligands as SMILES strings or CSV files to get BBB permeability predictions.
        The model uses RDKit to calculate molecular descriptors from structure and predicts BBB permeability
        using a sparse-label multi-task ensemble model.
        """
    )

    # Check if artifacts directory exists
    artifacts_dir = "artifacts"
    if not os.path.exists(artifacts_dir):
        st.warning(
            f"⚠️ **Model artifacts not found.**\n\n"
            f"The '{artifacts_dir}' directory is missing. Please ensure you have downloaded and placed "
            f"the following files:\n"
            f"- `{artifacts_dir}/descriptor_cols.json`\n"
            f"- `{artifacts_dir}/stage2_feature_cols.json`\n"
            f"- `{artifacts_dir}/models/stage1_efflux.joblib`\n"
            f"- `{artifacts_dir}/models/stage1_influx.joblib`\n"
            f"- `{artifacts_dir}/models/stage1_pampa.joblib`\n"
            f"- `{artifacts_dir}/models/stage1_cns.joblib`\n"
            f"- `{artifacts_dir}/models/stage2_bbb.joblib`\n\n"
            f"You can specify a different artifacts directory path if needed."
        )
        artifacts_dir = st.text_input("Artifacts directory path:", value="artifacts", key="artifacts_dir")

    predictor, error_msg = load_predictor(artifacts_dir)

    if error_msg:
        st.error(error_msg)
        st.info("**Note:** Without the model artifacts, you can still use this page to visualize molecular structures and compute RDKit descriptors, but predictions will not be available.")
        predictor = None
    else:
        st.success(f"✓ Model loaded successfully from `{artifacts_dir}`!")

    st.divider()

    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Single SMILES", "Multiple SMILES (one per line)", "CSV file upload"],
        horizontal=True,
        key="input_method"
    )

    smiles_list = []

    if input_method == "Single SMILES":
        smiles_input = st.text_input(
            "Enter SMILES string:",
            placeholder="e.g., CCO (ethanol) or CC(=O)O (acetic acid)",
            key="single_smiles"
        )
        if smiles_input:
            smiles_list = [smiles_input.strip()]
            
    elif input_method == "Multiple SMILES (one per line)":
        smiles_text = st.text_area(
            "Enter SMILES strings (one per line):",
            placeholder="CCO\nCC(=O)O\nC1=CC=CC=C1",
            height=150,
            key="multi_smiles"
        )
        if smiles_text:
            smiles_list = [s.strip() for s in smiles_text.strip().split("\n") if s.strip()]

    elif input_method == "CSV file upload":
        uploaded_file = st.file_uploader(
            "Upload CSV file:",
            type=["csv"],
            help="CSV file should contain a column named 'smiles' or 'SMILES' with SMILES strings.",
            key="csv_upload"
        )
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                # Find SMILES column (case-insensitive)
                smiles_col = None
                for col in df_upload.columns:
                    if col.lower() == "smiles":
                        smiles_col = col
                        break
                
                if smiles_col is None:
                    st.error(f"No 'smiles' column found in CSV. Available columns: {', '.join(df_upload.columns)}")
                else:
                    smiles_list = df_upload[smiles_col].astype(str).tolist()
                    st.success(f"✓ Loaded {len(smiles_list)} SMILES from CSV file")
                    
                    # Show preview
                    with st.expander("📋 Preview uploaded data"):
                        st.dataframe(df_upload.head(10), use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

    # Process and validate SMILES
    if smiles_list:
        st.divider()
        
        # Canonicalize and validate
        with st.spinner("Validating and canonicalizing SMILES..."):
            canon_smiles = []
            invalid_indices = []
            
            for idx, smi in enumerate(smiles_list):
                canon = canonicalize_smiles(smi)
                if canon is None:
                    invalid_indices.append(idx)
                else:
                    canon_smiles.append((idx, smi, canon))
        
        # Show validation results
        if invalid_indices:
            st.warning(f"⚠️ {len(invalid_indices)} invalid SMILES string(s) found and will be skipped.")
            with st.expander("❌ Invalid SMILES", expanded=False):
                for idx in invalid_indices:
                    st.text(f"Row {idx + 1}: {smiles_list[idx]}")
        
        if canon_smiles:
            valid_smiles = [canon for _, _, canon in canon_smiles]
            
            st.success(f"✓ {len(valid_smiles)} valid SMILES string(s) ready for prediction")
            
            # Show molecule previews (only if Draw module is available)
            if len(valid_smiles) <= 20 and DRAW_AVAILABLE:  # Only show previews for small batches
                st.subheader("Molecular Structure Preview")
                cols_per_row = 4
                rows = (len(valid_smiles) + cols_per_row - 1) // cols_per_row
                
                for row_idx in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx, col in enumerate(cols):
                        mol_idx = row_idx * cols_per_row + col_idx
                        if mol_idx < len(valid_smiles):
                            try:
                                mol = Chem.MolFromSmiles(valid_smiles[mol_idx])
                                if mol and Draw is not None:
                                    img = Draw.MolToImage(mol, size=(300, 300))
                                    col.image(img, caption=valid_smiles[mol_idx][:50], use_container_width=True)
                            except Exception as e:
                                col.error(f"Error rendering: {e}")
            elif not DRAW_AVAILABLE:
                st.info("ℹ️ Molecular structure visualization is not available on this platform. Descriptor computation and predictions will work normally.")
            
            # Compute descriptors
            if st.button("🚀 Compute Descriptors & Make Predictions", type="primary"):
                with st.spinner("Computing RDKit descriptors and making predictions..."):
                    try:
                        # Compute descriptors
                        desc_df = compute_rdkit_descriptors(valid_smiles)
                        
                        st.subheader("📊 Descriptor Information")
                        st.info(f"✓ Computed {len(desc_df.columns)} RDKit descriptors for {len(valid_smiles)} molecule(s)")
                        
                        # Show descriptor summary
                        with st.expander("📈 Descriptor Summary Statistics", expanded=False):
                            st.dataframe(desc_df.describe(), use_container_width=True)
                        
                        # Make predictions if model is available
                        if predictor is None:
                            st.warning("⚠️ Predictions unavailable - model artifacts not loaded. Descriptors computed successfully.")
                        else:
                            # Make predictions
                            predictions = predictor.predict_proba(valid_smiles)
                            
                            # Display results
                            st.subheader("🎯 BBB Permeability Predictions")
                            
                            # Add threshold slider
                            threshold = st.slider(
                                "Classification threshold:",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.5,
                                step=0.01,
                                help="Molecules with BBB probability ≥ threshold are classified as BBB+",
                                key="threshold"
                            )
                            
                            # Add binary predictions
                            predictions["BBB+"] = (predictions["p_bbb_plus"] >= threshold).astype(int)
                            predictions["BBB-"] = ((predictions["p_bbb_plus"] < threshold).astype(int))
                            
                            # Rename columns for better display
                            display_cols = {
                                "smiles": "SMILES",
                                "p_bbb_plus": "BBB Permeability Probability",
                                "p_efflux_mech": "Efflux Mechanism",
                                "p_influx_mech": "Influx Mechanism",
                                "p_pampa_mech": "PAMPA Mechanism",
                                "p_cns_mech": "CNS Mechanism",
                                "BBB+": "BBB+",
                                "BBB-": "BBB-"
                            }
                            display_df = predictions.rename(columns=display_cols)[list(display_cols.values())]
                            
                            # Format probability columns
                            prob_cols = [c for c in display_df.columns if "Probability" in c or "Mechanism" in c]
                            for col in prob_cols:
                                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                            
                            # Display results table
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Molecules", len(predictions))
                            with col2:
                                bbb_plus_count = (predictions["p_bbb_plus"] >= threshold).sum()
                                st.metric("BBB+ (Predicted)", bbb_plus_count, f"{(bbb_plus_count/len(predictions)*100):.1f}%")
                            with col3:
                                st.metric("Avg BBB Probability", f"{predictions['p_bbb_plus'].mean():.4f}")
                            with col4:
                                st.metric("Max BBB Probability", f"{predictions['p_bbb_plus'].max():.4f}")
                            
                            # Interpretation guide
                            with st.expander("📖 Interpretation Guide", expanded=False):
                                st.markdown("""
                                **BBB Permeability Probability (p_bbb_plus):**
                                - **0.0 - 0.3:** Low probability of BBB permeability
                                - **0.3 - 0.7:** Moderate probability of BBB permeability
                                - **0.7 - 1.0:** High probability of BBB permeability
                                
                                **Mechanism Probabilities:**
                                - **p_efflux_mech:** Probability of efflux transporter involvement
                                - **p_influx_mech:** Probability of influx transporter involvement
                                - **p_pampa_mech:** Probability of passive membrane permeability (PAMPA)
                                - **p_cns_mech:** Probability of CNS-related mechanisms
                                
                                The default threshold of 0.5 is commonly used, but you may adjust it based on your application.
                                According to the manuscript, the Youden threshold (≈0.793) may provide better classification performance.
                                """)
                            
                            # Download results
                            st.subheader("💾 Download Results")
                            
                            # Prepare download dataframe
                            download_df = predictions.copy()
                            download_df["bbb_pred"] = (predictions["p_bbb_plus"] >= threshold).astype(int)
                            
                            csv = download_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="bbb_predictions.csv",
                                mime="text/csv",
                                key="download_csv"
                            )
                            
                            # Show descriptors used (if available)
                            if hasattr(predictor, 'descriptor_cols'):
                                with st.expander("🔬 Descriptors Used by Model", expanded=False):
                                    st.info(f"Model uses {len(predictor.descriptor_cols)} descriptor(s)")
                                    st.write("First 20 descriptors:")
                                    st.code(", ".join(predictor.descriptor_cols[:20]))
                                
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        with st.expander("Error details", expanded=False):
                            st.code(traceback.format_exc())
        else:
            st.warning("No valid SMILES strings to process.")
    else:
        st.info("👆 Please enter SMILES strings or upload a CSV file to get started.")

    # Footer
    st.divider()
    st.markdown(
        """
        **Note:** This interface uses RDKit to compute molecular descriptors directly from SMILES structures.
        The model is trained on RDKit descriptors that can be computed from molecular structure alone.
        For best results, ensure your SMILES strings are valid and properly formatted.
        """
    )


# ============================================================================
# MAIN APP - NAVIGATION
# ============================================================================

def main():
    """Main app entry point with navigation."""
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["🏠 Home", "📚 Documentation", "🧪 Ligand Prediction"],
        key="page_selector"
    )

    # Render selected page
    if page == "🏠 Home":
        render_home_page()
    elif page == "📚 Documentation":
        render_documentation_page()
    elif page == "🧪 Ligand Prediction":
        render_ligand_prediction_page()


if __name__ == "__main__":
    main()
