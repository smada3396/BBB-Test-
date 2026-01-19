"""
Script to download files from Google Drive.
Supports both public files and files requiring authentication.
"""

import os
import sys
from pathlib import Path
from typing import Optional

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import pickle
    import io
    HAS_API = True
except ImportError:
    HAS_API = False
    print("Note: Google Drive API libraries not installed. Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")


def download_file_public(file_id: str, output_path: str, quiet: bool = False) -> bool:
    """
    Download a public file from Google Drive using gdown.
    
    Args:
        file_id: Google Drive file ID (from the shareable link)
        output_path: Where to save the file
        quiet: Whether to suppress output
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        gdown.download(url, output_path, quiet=quiet)
        
        if os.path.exists(output_path):
            print(f"✓ Downloaded: {output_path}")
            return True
        else:
            print(f"✗ Failed to download: {output_path}")
            return False
    except Exception as e:
        print(f"✗ Error downloading {file_id}: {e}")
        return False


def download_folder_public(folder_id: str, output_dir: str, quiet: bool = False) -> int:
    """
    Download all files from a public Google Drive folder.
    
    Args:
        folder_id: Google Drive folder ID
        output_dir: Directory to save files
        quiet: Whether to suppress output
    
    Returns:
        Number of files successfully downloaded
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        downloaded = gdown.download_folder(url, output=output_dir, quiet=quiet, use_cookies=False)
        print(f"✓ Downloaded {len(downloaded)} file(s) to {output_dir}")
        return len(downloaded)
    except Exception as e:
        print(f"✗ Error downloading folder {folder_id}: {e}")
        return 0


def get_drive_service(credentials_file: str = "credentials.json", token_file: str = "token.pickle"):
    """
    Authenticate and return Google Drive API service.
    
    Args:
        credentials_file: Path to OAuth2 credentials JSON file
        token_file: Path to save/load authentication token
    
    Returns:
        Google Drive service object
    """
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None
    
    # Load existing token
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_file):
                raise FileNotFoundError(
                    f"Credentials file not found: {credentials_file}\n"
                    "Please download it from Google Cloud Console:\n"
                    "1. Go to https://console.cloud.google.com/\n"
                    "2. Create a project or select existing\n"
                    "3. Enable Google Drive API\n"
                    "4. Create OAuth 2.0 credentials\n"
                    "5. Download credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save token for future use
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)


def download_file_with_api(file_id: str, output_path: str, service) -> bool:
    """
    Download a file using Google Drive API (supports private files).
    
    Args:
        file_id: Google Drive file ID
        output_path: Where to save the file
        service: Authenticated Google Drive service
    
    Returns:
        True if successful
    """
    try:
        request = service.files().get_media(fileId=file_id)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download progress: {int(status.progress() * 100)}%")
        
        print(f"✓ Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {file_id}: {e}")
        return False


def download_folder_with_api(folder_id: str, output_dir: str, service) -> int:
    """
    Download all files from a folder using Google Drive API.
    
    Args:
        folder_id: Google Drive folder ID
        output_dir: Directory to save files
        service: Authenticated Google Drive service
    
    Returns:
        Number of files downloaded
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # List all files in folder
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id, name, mimeType)"
        ).execute()
        
        items = results.get('files', [])
        count = 0
        
        for item in items:
            file_id = item['id']
            file_name = item['name']
            file_path = os.path.join(output_dir, file_name)
            
            if download_file_with_api(file_id, file_path, service):
                count += 1
        
        print(f"✓ Downloaded {count} file(s) to {output_dir}")
        return count
    except Exception as e:
        print(f"✗ Error downloading folder {folder_id}: {e}")
        return 0


def extract_file_id(url: str) -> Optional[str]:
    """
    Extract file/folder ID from various Google Drive URL formats.
    
    Supports:
    - https://drive.google.com/file/d/FILE_ID/view
    - https://drive.google.com/open?id=FILE_ID
    - https://drive.google.com/uc?id=FILE_ID
    - https://drive.google.com/drive/folders/FOLDER_ID
    - Direct file/folder ID
    """
    if '/' in url:
        # Extract ID from URL
        if '/file/d/' in url:
            return url.split('/file/d/')[1].split('/')[0]
        elif '/drive/folders/' in url:
            return url.split('/drive/folders/')[1].split('/')[0].split('?')[0]
        elif 'id=' in url:
            return url.split('id=')[1].split('&')[0]
        elif 'uc?id=' in url:
            return url.split('uc?id=')[1].split('&')[0]
    else:
        # Assume it's already a file ID
        return url
    return None


def main():
    """Main function to handle command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download files from Google Drive')
    parser.add_argument('url_or_id', help='Google Drive URL or file/folder ID')
    parser.add_argument('-o', '--output', help='Output path or directory', default='.')
    parser.add_argument('-a', '--api', action='store_true', help='Use Google Drive API (for private files)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')
    parser.add_argument('--credentials', default='credentials.json', help='Path to credentials.json (for API)')
    
    args = parser.parse_args()
    
    # Extract file/folder ID
    file_or_folder_id = extract_file_id(args.url_or_id)
    
    if not file_or_folder_id:
        print("Error: Could not extract file/folder ID from URL")
        return
    
    if args.api:
        if not HAS_API:
            print("Error: Google Drive API libraries not installed.")
            print("Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
            return
        
        try:
            service = get_drive_service(args.credentials)
            
            # Try as file first, then as folder
            try:
                file_metadata = service.files().get(fileId=file_or_folder_id, fields='mimeType').execute()
                mime_type = file_metadata.get('mimeType', '')
                
                if mime_type == 'application/vnd.google-apps.folder':
                    download_folder_with_api(file_or_folder_id, args.output, service)
                else:
                    output_path = args.output if os.path.dirname(args.output) else os.path.join(args.output, 'download')
                    download_file_with_api(file_or_folder_id, output_path, service)
            except Exception as e:
                print(f"Error: {e}")
        except Exception as e:
            print(f"Authentication error: {e}")
    else:
        # Try public download (assume it's a file first)
        output_path = args.output
        if os.path.isdir(output_path) or not os.path.dirname(output_path):
            output_path = os.path.join(output_path, 'download')
        
        if download_file_public(file_or_folder_id, output_path, args.quiet):
            print("Note: If download failed, the file might be private. Use --api flag with credentials.")
        else:
            # Try as folder
            print("Trying as folder...")
            download_folder_public(file_or_folder_id, args.output, args.quiet)


if __name__ == "__main__":
    main()
