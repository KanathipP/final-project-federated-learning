from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm

# ===============================
# CONFIGURATION
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ZIP_FILE = BASE_DIR / "BCICIV_2a_gdf.zip"
URL = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"


# ===============================
# HELPERS
# ===============================
class DownloadProgressBar(tqdm):
    """Progress bar for urllib.urlretrieve"""
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


# ===============================
# CORE FUNCTIONS
# ===============================
def download_file(url: str, dest: Path) -> None:
    """Download file if it doesn't already exist."""
    tag = "[Download]"
    if dest.exists():
        print(f"{tag} Skip: {dest.name} already exists")
        return

    print(f"{tag} Fetching from {url}")
    with DownloadProgressBar(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=dest.name
    ) as progress:
        urllib.request.urlretrieve(url, dest, reporthook=progress.update_to)
    print(f"{tag} Done: saved to {dest}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file if target directory not present."""
    tag = "[Extract]"
    if extract_to.exists():
        print(f"{tag} Skip: already extracted at {extract_to}")
        return

    print(f"{tag} Extracting {zip_path.name} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_to)
    print(f"{tag} Done: extracted to {extract_to}")


def cleanup_file(path: Path) -> None:
    """Remove file safely."""
    tag = "[Cleanup]"
    if path.exists():
        path.unlink()
        print(f"{tag} Removed: {path.name}")
    else:
        print(f"{tag} Skip: {path.name} not found")


# ===============================
# MAIN PIPELINE
# ===============================
def download_and_extract_data(delete_zip: bool = False) -> None:
    """Full pipeline: download → extract → optional cleanup"""
    tag = "[Pipeline]"
    print(f"{tag} Starting data preparation...")
    DATA_DIR.mkdir(exist_ok=True)
    download_file(URL, ZIP_FILE)
    extract_zip(ZIP_FILE, DATA_DIR)
    if delete_zip:
        cleanup_file(ZIP_FILE)
    print(f"{tag} Completed.")


if __name__ == "__main__":
    download_and_extract_data(delete_zip=False)
