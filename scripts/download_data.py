import zipfile
from pathlib import Path
import shutil

"""Download PCam dataset."""

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


def download_and_unzip():
    data_dir = ensure_data_dir()
    print("Downloading PCam dataset...")
    path = download('histopathologic-cancer-detection')
    print(f"Downloaded to: {path}")

    # Move files to data_dir
    downloaded_dir = Path(path)
    for file in downloaded_dir.iterdir():
        if file.is_file():
            shutil.move(str(file), str(data_dir / file.name))

    # Unzip
    zip_files = ["train.zip", "test.zip"]
    for zip_file in zip_files:
        zip_path = data_dir / zip_file
        if zip_path.exists():
            print(f"Unzipping {zip_file}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_path.unlink()
            print(f"Extracted and removed {zip_file}")

    print("Download and extraction complete.")


if __name__ == "__main__":
    download_and_unzip()
