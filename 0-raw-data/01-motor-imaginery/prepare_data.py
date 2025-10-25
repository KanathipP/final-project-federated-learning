import os
import urllib.request
import zipfile
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ZIP_PATH = os.path.join(BASE_DIR, "BCICIV_2a_gdf.zip")
EXTRACT_DIR = os.path.join(BASE_DIR, "data")

URL = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"


class TqdmUpTo(tqdm):
    """tqdm wrapper for urllib.urlretrieve reporthook"""
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_if_needed():
    if os.path.exists(ZIP_PATH):
        print("[skip] zip already downloaded")
        return

    print("[download] fetching zip...")
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=os.path.basename(ZIP_PATH)) as t:
        urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=t.update_to)
    print(f"[done] downloaded to {ZIP_PATH}")


def extract_if_needed():
    if os.path.exists(EXTRACT_DIR):
        print("[skip] already extracted")
        return

    print("[extract] extracting zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"[done] extracted to {EXTRACT_DIR}")


def cleanup_zip(optional_delete=True):
    if optional_delete and os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("[clean] removed zip")


def main():
    download_if_needed()
    extract_if_needed()
    cleanup_zip(optional_delete=False)


if __name__ == "__main__":
    main()
