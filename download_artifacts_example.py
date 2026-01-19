"""
Example script to download model artifacts from Google Drive.
Modify the file/folder IDs or URLs based on your actual Google Drive links.
"""

from download_from_gdrive import (
    download_file_public,
    download_folder_public,
    extract_file_id,
    get_drive_service,
    download_folder_with_api
)
import os

# Configuration
ARTIFACTS_DIR = "artifacts"
USE_API = False  # Set to True if files are private and you have credentials.json

# Option 1: Download from a folder (recommended)
# Replace with your actual Google Drive folder URL or ID
FOLDER_URL_OR_ID = ""  # e.g., "https://drive.google.com/drive/folders/1abc123..." or "1abc123..."

# Option 2: Download individual files
# Dictionary mapping file paths to Google Drive file IDs or URLs
FILES_TO_DOWNLOAD = {
    # Replace with your actual file IDs/URLs
    f"{ARTIFACTS_DIR}/descriptor_cols.json": "",
    f"{ARTIFACTS_DIR}/stage2_feature_cols.json": "",
    f"{ARTIFACTS_DIR}/models/stage1_efflux.joblib": "",
    f"{ARTIFACTS_DIR}/models/stage1_influx.joblib": "",
    f"{ARTIFACTS_DIR}/models/stage1_pampa.joblib": "",
    f"{ARTIFACTS_DIR}/models/stage1_cns.joblib": "",
    f"{ARTIFACTS_DIR}/models/stage2_bbb.joblib": "",
}


def download_artifacts_from_folder(folder_url_or_id: str, use_api: bool = False):
    """Download all artifacts from a Google Drive folder."""
    folder_id = extract_file_id(folder_url_or_id)
    
    if not folder_id:
        print("Error: Could not extract folder ID from URL")
        return
    
    if use_api:
        print("Using Google Drive API (for private files)...")
        service = get_drive_service()
        count = download_folder_with_api(folder_id, ARTIFACTS_DIR, service)
        print(f"Downloaded {count} file(s)")
    else:
        print("Downloading public folder...")
        count = download_folder_public(folder_id, ARTIFACTS_DIR)
        print(f"Downloaded {count} file(s)")


def download_artifacts_individually(files_dict: dict, use_api: bool = False):
    """Download artifacts individually by file."""
    service = None
    if use_api:
        print("Using Google Drive API (for private files)...")
        service = get_drive_service()
    
    success_count = 0
    for output_path, file_url_or_id in files_dict.items():
        if not file_url_or_id:
            print(f"Skipping {output_path} (no file ID/URL provided)")
            continue
        
        file_id = extract_file_id(file_url_or_id)
        if not file_id:
            print(f"Error: Could not extract file ID from {file_url_or_id}")
            continue
        
        if use_api and service:
            from download_from_gdrive import download_file_with_api
            if download_file_with_api(file_id, output_path, service):
                success_count += 1
        else:
            if download_file_public(file_id, output_path):
                success_count += 1
    
    print(f"Successfully downloaded {success_count} out of {len(files_dict)} file(s)")


if __name__ == "__main__":
    print("=" * 60)
    print("BBB Model Artifacts Downloader")
    print("=" * 60)
    
    # Check if folder URL/ID is provided
    if FOLDER_URL_OR_ID:
        print("\nDownloading from folder...")
        download_artifacts_from_folder(FOLDER_URL_OR_ID, USE_API)
    else:
        # Check if any individual file URLs/IDs are provided
        has_files = any(file_id for file_id in FILES_TO_DOWNLOAD.values())
        if has_files:
            print("\nDownloading individual files...")
            download_artifacts_individually(FILES_TO_DOWNLOAD, USE_API)
        else:
            print("\n" + "=" * 60)
            print("INSTRUCTIONS:")
            print("=" * 60)
            print("\n1. For folder download:")
            print("   - Set FOLDER_URL_OR_ID to your Google Drive folder URL or ID")
            print("   - Example: FOLDER_URL_OR_ID = 'https://drive.google.com/drive/folders/1abc123...'")
            print("\n2. For individual file download:")
            print("   - Fill in the FILES_TO_DOWNLOAD dictionary with file IDs/URLs")
            print("   - Each value should be a Google Drive file URL or ID")
            print("\n3. For private files:")
            print("   - Set USE_API = True")
            print("   - Download credentials.json from Google Cloud Console")
            print("   - Place credentials.json in the same directory as this script")
            print("\n4. Run the script:")
            print("   python download_artifacts_example.py")
            print("\n" + "=" * 60)
