import subprocess
import sys

# List of packages required by both scripts
required_packages = [
    "pandas",
    "joblib",
    "yt-dlp",
    "soundfile",
    "numpy"
]

for package in required_packages:
    try:
        __import__(package)
        print(f"[SKIP] {package} is already installed.")
    except ImportError:
        print(f"[INSTALL] Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from audioset_downloader import Downloader

# Create downloader (auto-refresh cookies from Chrome)
downloader = Downloader(
    root_path="./balanced",  # change to your path
    n_jobs=2,
    download_type="balanced_train",
    labels=None,
    cookies_from_browser=True
)

downloader.download(format="vorbis", quality=5)

