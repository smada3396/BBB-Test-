"""
Script to generate BBB model artifacts by training the models.

This script will train the two-stage BBB permeability prediction model
and save all required artifacts to the artifacts/ directory.

Requirements:
- Training data CSV files with 'smiles' and 'label' columns:
  - bbb_internal.csv (BBB internal dataset)
  - efflux.csv (efflux mechanism dataset)
  - influx.csv (influx mechanism dataset)
  - pampa.csv (PAMPA dataset)
  - cns.csv (CNS dataset)
  - bbbp_external.csv (optional, for external validation)
"""

import os
import sys

# Add the current directory to path to import bbb_model_train
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from bbb_model_train import TrainPaths, train_two_stage
except ImportError:
    print("Error: Could not import bbb_model_train.")
    print("Make sure bbb_model_train.py is in the same directory.")
    sys.exit(1)


def main():
    """Generate model artifacts by training the models."""
    
    print("=" * 70)
    print("BBB Model Artifact Generator")
    print("=" * 70)
    print()
    
    # Get data directory path
    data_dir = input("Enter path to directory containing training CSV files (default: 'data'): ").strip()
    if not data_dir:
        data_dir = "data"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please create the directory and add your training CSV files with 'smiles' and 'label' columns.")
        sys.exit(1)
    
    # Define paths to training data
    paths = TrainPaths(
        bbb_internal=os.path.join(data_dir, "bbb_internal.csv"),
        efflux=os.path.join(data_dir, "efflux.csv"),
        influx=os.path.join(data_dir, "influx.csv"),
        pampa=os.path.join(data_dir, "pampa.csv"),
        cns=os.path.join(data_dir, "cns.csv"),
        bbbp_external=os.path.join(data_dir, "bbbp_external.csv") if os.path.exists(os.path.join(data_dir, "bbbp_external.csv")) else None,
    )
    
    # Check which files exist
    required_files = [
        ("bbb_internal", paths.bbb_internal),
        ("efflux", paths.efflux),
        ("influx", paths.influx),
        ("pampa", paths.pampa),
        ("cns", paths.cns),
    ]
    
    missing_files = []
    for name, path in required_files:
        if not os.path.exists(path):
            missing_files.append(f"  - {name}: {path}")
    
    if missing_files:
        print("Error: The following required training data files are missing:")
        for f in missing_files:
            print(f)
        print()
        print("Each CSV file must contain columns: 'smiles' and 'label'")
        sys.exit(1)
    
    print("Training data files found:")
    for name, path in required_files:
        print(f"  ✓ {name}: {path}")
    if paths.bbbp_external:
        print(f"  ✓ bbbp_external (optional): {paths.bbbp_external}")
    print()
    
    # Get artifacts directory
    artifacts_dir = input("Enter artifacts output directory (default: 'artifacts'): ").strip()
    if not artifacts_dir:
        artifacts_dir = "artifacts"
    
    # Get hyperparameter settings
    print()
    print("Training hyperparameters:")
    n_trials = input("Number of hyperparameter optimization trials (default: 100): ").strip()
    n_trials = int(n_trials) if n_trials.isdigit() else 100
    
    n_splits = input("Number of CV folds (default: 5): ").strip()
    n_splits = int(n_splits) if n_splits.isdigit() else 5
    
    seed = input("Random seed (default: 42): ").strip()
    seed = int(seed) if seed.isdigit() else 42
    
    print()
    print("=" * 70)
    print("Starting model training...")
    print("This may take a while (hyperparameter optimization + model training)")
    print("=" * 70)
    print()
    
    try:
        # Train the model
        train_two_stage(
            paths,
            artifacts_dir=artifacts_dir,
            seed=seed,
            n_splits=n_splits,
            n_trials=n_trials,
        )
        
        print()
        print("=" * 70)
        print("SUCCESS! Model artifacts have been generated.")
        print("=" * 70)
        print()
        print(f"Artifacts saved to: {artifacts_dir}/")
        print()
        print("Generated files:")
        print(f"  - {artifacts_dir}/descriptor_cols.json")
        print(f"  - {artifacts_dir}/stage2_feature_cols.json")
        print(f"  - {artifacts_dir}/models/stage1_efflux.joblib")
        print(f"  - {artifacts_dir}/models/stage1_influx.joblib")
        print(f"  - {artifacts_dir}/models/stage1_pampa.joblib")
        print(f"  - {artifacts_dir}/models/stage1_cns.joblib")
        print(f"  - {artifacts_dir}/models/stage2_bbb.joblib")
        print()
        print("You can now use these artifacts with the Streamlit app!")
        
    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR: Training failed!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
