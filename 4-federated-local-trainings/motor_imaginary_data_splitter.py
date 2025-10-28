import sys
from pathlib import Path
import importlib
import shutil

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(BASE_DIR / "../0-raw-data/motor-imaginary")
EXTRACT_DIR = Path(DATASET_DIR / "data")

def download_and_extract_motor_imaginary_data():
    """Ensure that the raw Motor Imaginary dataset is downloaded and extracted."""
    target_dir = DATASET_DIR.resolve()
    if str(target_dir) not in sys.path:
        sys.path.append(str(target_dir))

    import data_fetcher
    importlib.reload(data_fetcher)

    data_fetcher.download_and_extract_data(delete_zip=False)

def copy_subject_files(app_root: Path):
    """Copy A0{i}{E,T}.gdf files into app_root/data/{i}/"""
    for subj in range(1, 10):  # subjects 1..9
        subj_dir = app_root / "data" / f"{subj}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        for session_suffix in ["E", "T"]:
            filename = f"A0{subj}{session_suffix}.gdf"
            src = EXTRACT_DIR / filename
            dst = subj_dir / filename

            if src.exists():
                shutil.copy2(src, dst)
                print(f"[OK] copied {src.name} -> {subj_dir}")
            else:
                print(f"[WARN] missing {src}, skipped")

def organize_motor_imaginary_data(app_name: str):
    """
    1. Make sure data is downloaded
    2. Create {app_name}/data/{1..9}/
    3. Copy corresponding raw .gdf files
    """
    # Step 1
    download_and_extract_motor_imaginary_data()

    # Step 2
    app_root = (BASE_DIR / app_name).resolve()
    app_root.mkdir(parents=True, exist_ok=True)

    # Step 3
    copy_subject_files(app_root)

    print(f"✅ Done organizing raw data under: {app_root}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize raw Motor Imaginary data for a specific FLWR app"
    )
    parser.add_argument(
        "app_name",
        type=str,
        help="FLWR app name (used as folder name, e.g. 'flwr_motor_imagery')",
    )
    args = parser.parse_args()

    organize_motor_imaginary_data(args.app_name)
