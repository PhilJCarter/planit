import os
import pathlib
import shutil
import urllib.request
import zipfile
import platformdirs

DATA_VERSION = "v1.2.4"

CACHE_DIR = platformdirs.user_cache_path("planit")
#CACHE_DIR = pathlib.Path.home() / ".planit"
VERSION_FILE = CACHE_DIR / "data_version.txt"

# Map directory keys to their remote zip file names
REMOTE_DIRECTORIES = {
    "aneos-pyrolite-2022": "aneos-pyrolite-2022.zip",
    "aqua-water": "aqua-water.zip",
    "5-phase-water": "5-phase-water.zip",
    "aneos-Fe85Si15-2020-master": "aneos-Fe85Si15-2020-master.zip"
}

REPO_SOURCE_DATA = {
    "aneos-pyrolite-2022": pathlib.Path(__file__).parent.parent / "eos/data/aneos-pyrolite-2022",
    "aqua-water": pathlib.Path(__file__).parent.parent / "eos/data/aqua-water",
    "5-phase-water": pathlib.Path(__file__).parent.parent / "eos/data/5-phase-water",
    "aneos-Fe85Si15-2020-master": pathlib.Path(__file__).parent.parent / "eos/data/aneos-Fe85Si15-2020-master"}


def check_local_version_valid() -> bool:
    """Checks if the downloaded data match the required version."""
    if not VERSION_FILE.exists():
        return False
    try:
        return VERSION_FILE.read_text().strip() == DATA_VERSION
    except Exception:
        return False


def get_eosdata_path(dir_key: str, use_repo_source=True, debug=False) -> pathlib.Path:
    """
    Returns the path to a local directory.
    Downloads and extracts the entire directory structure if missing or outdated.
    """
    if (dir_key not in REMOTE_DIRECTORIES) and (dir_key not in REPO_SOURCE_DATA):
        raise ValueError(f"Unknown EoS key: '{dir_key}'")

    source_repo_path = REPO_SOURCE_DATA.get(dir_key)
    if source_repo_path and source_repo_path.exists() and use_repo_source:
        return str(source_repo_path)+'/'

    zip_filename = REMOTE_DIRECTORIES[dir_key]
    folder_name = zip_filename.replace(".zip", "") # The extracted folder name
    local_dir_path = CACHE_DIR / folder_name

    # Check versioning status
    is_version_valid = check_local_version_valid()

    # Trigger update: Clear out the old directory if it's outdated
    if not is_version_valid and local_dir_path.exists():
        print(f"🔄 Outdated data found. Removing old version of {folder_name}...")
        shutil.rmtree(local_dir_path, ignore_errors=True)

    # Standard Lazy Load Check: If it exists and matches version, return it
    if is_version_valid and local_dir_path.exists():
        return str(local_dir_path)+'/'

    # Construct the download URL
    url = f"https://github.com/PhilJCarter/planit/releases/download/{DATA_VERSION}/{zip_filename}"
    temp_zip_path = CACHE_DIR / f"temp_{zip_filename}"

    print(f"📥 Fetching EoS files: '{dir_key}' ({DATA_VERSION})...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Download the single zip file
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(temp_zip_path, 'wb') as out_file:
            out_file.write(response.read())

        # Step 2: Unzip the entire directory structure automatically
        if debug:
            print(f"📦 Extracting files into {local_dir_path}...")
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(CACHE_DIR) # Extracts the folder directly into your cache

        # Step 3: Write the version token file and clean up the zip archive
        VERSION_FILE.write_text(DATA_VERSION)
        temp_zip_path.unlink() # Delete the temporary zip archive to preserve disk space
        
        if debug:
            print(f"✅ Directory {folder_name} successfully set up!")
        return str(local_dir_path)+'/'

    except Exception as e:
        # Clean up corrupted states on failure
        if temp_zip_path.exists():
            temp_zip_path.unlink()
            
        raise RuntimeError(f"Failed to fetch directory bundle '{dir_key}' from GitHub: {e}")
