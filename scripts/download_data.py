import os
import subprocess

"""Download PCam dataset using Kaggle API."""

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
KAGGLE_COMPETITION = "histopathologic-cancer-detection"


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def download_dataset():
    data_dir = ensure_data_dir()
    print(f"Downloading PCam dataset into: {data_dir}")
    subprocess.run([
        "kaggle",
        "competitions",
        "download",
        "-c",
        KAGGLE_COMPETITION,
        "-p",
        data_dir,
        "--force",
    ], check=True)
    print("Download complete. Extract files manually if needed.")


if __name__ == "__main__":
    download_dataset()
