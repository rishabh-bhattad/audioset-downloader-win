import os
import time
import joblib
import pandas as pd
import subprocess
import json


class Downloader:
    """
    This class implements the download of the AudioSet dataset.
    It only downloads the audio files according to the provided list of labels and associated timestamps.
    """

    def __init__(self, 
                 root_path: str,
                 labels: list = None,  # None to download all the dataset
                 n_jobs: int = 1,
                 download_type: str = 'unbalanced_train',
                 copy_and_replicate: bool = True,
                 cookie_file: str = None,  # Path to a cookie file for yt-dlp, if needed
                 cookies_from_browser: bool = False,  # If True, cookies will be extracted from the browser
                 max_retries: int = 5,  # Number of retries for failed downloads
                 retry_delay: int = 5,  # Seconds to wait between retries
                 start_idx: int = None,  # Optional start index for chunked download
                 end_idx: int = None     # Optional end index for chunked download
                 ):
        self.root_path = root_path
        self.labels = labels
        self.n_jobs = n_jobs
        self.download_type = download_type
        self.copy_and_replicate = copy_and_replicate
        self.cookie_file = cookie_file
        self.cookies_from_browser = cookies_from_browser
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.start_idx = start_idx
        self.end_idx = end_idx

        os.makedirs(self.root_path, exist_ok=True)
        self.read_class_mapping()

    def read_class_mapping(self):
        class_df = pd.read_csv(
            f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv", 
            sep=',',
        )
        self.display_to_machine_mapping = dict(zip(class_df['display_name'], class_df['mid']))
        self.machine_to_display_mapping = dict(zip(class_df['mid'], class_df['display_name']))
        return

    def get_audio_duration(self, file_path):
        try:
            ffprobe_path = os.path.join("bin", "ffprobe.exe")

            result = subprocess.run([
                ffprobe_path, "-v", "quiet", "-show_entries", "format=duration",
                "-of", "json", file_path
            ], capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration
        except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError):
            return None

    def _build_ytdlp_command(self, ytid: str, file_path: str, start_seconds: float, end_seconds: float):
        base_command = (f'yt-dlp -x --audio-format {self.format} --audio-quality {self.quality} '
                        f'--ffmpeg-location "./bin" '
                        f'--output "{file_path}" --postprocessor-args "-ss {start_seconds} -to {end_seconds}" '
                        f'https://www.youtube.com/watch?v={ytid}')

        if self.cookie_file and self.cookies_from_browser:
            print(f"[WARNING] Both cookie_file and cookies_from_browser are set. Using cookie_file: {self.cookie_file}")
            base_command += f' --cookies "{self.cookie_file}"'
        elif self.cookies_from_browser:
            print("[INFO] Extracting cookies from browser...")
            base_command += ' --cookies-from-browser firefox'
        elif self.cookie_file:
            print(f"[INFO] Using cookie file: {self.cookie_file}")
            base_command += f' --cookies "{self.cookie_file}"'

        return base_command

    def download(self, format: str = 'vorbis', quality: int = 5):
        self.format = format
        self.quality = quality

        metadata = pd.read_csv(
            f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{self.download_type}_segments.csv", 
            sep=', ', 
            skiprows=3,
            header=None,
            names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
            engine='python'
        )

        if self.labels is not None:
            self.real_labels = [self.display_to_machine_mapping[label] for label in self.labels]
            metadata = metadata[metadata['positive_labels'].apply(lambda x: any([label in x for label in self.real_labels]))]

        metadata['positive_labels'] = metadata['positive_labels'].apply(lambda x: x.replace('"', ''))
        metadata = metadata.reset_index(drop=True)

        # Apply chunk indices
        if self.start_idx is not None:
            metadata = metadata.iloc[self.start_idx:]
        if self.end_idx is not None:
            metadata = metadata.iloc[:self.end_idx]

        print(f'Downloading {len(metadata)} files...')

        joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.download_file)(
                i,
                metadata.loc[i, 'YTID'], 
                metadata.loc[i, 'start_seconds'], 
                metadata.loc[i, 'end_seconds'], 
                metadata.loc[i, 'positive_labels'],
                len(metadata)
            ) for i in range(len(metadata))
        )

        print('Done.')

    def download_file(self, idx, ytid: str, start_seconds: float, end_seconds: float, positive_labels: str, total_rows: int):
        print("-"*160)
        print(f"[INFO] Downloading row {idx + 1} of {total_rows}...")  # show row and total
        if self.copy_and_replicate:
            for label in positive_labels.split(','):
                display_label = self.machine_to_display_mapping[label]
                os.makedirs(os.path.join(self.root_path, display_label), exist_ok=True)
        else:
            display_label = self.machine_to_display_mapping[positive_labels.split(',')[0]]
            os.makedirs(os.path.join(self.root_path, display_label), exist_ok=True)

        first_display_label = self.machine_to_display_mapping[positive_labels.split(',')[0]]
        ext = {'vorbis':'ogg','wav':'wav','mp3':'mp3','flac':'flac','opus':'opus','m4a':'m4a'}.get(self.format, self.format)
        file_path = os.path.join(self.root_path, first_display_label, f"{ytid}_{start_seconds}-{end_seconds}.{ext}")

        # skip if already exists and valid
        if os.path.exists(file_path):
            duration = self.get_audio_duration(file_path)
            if duration and duration > 0.0:
                print(f"[SKIP] Already downloaded: {file_path}")
                return
            else:
                print(f"[RETRY] Corrupted file, re-downloading: {file_path}")
                os.remove(file_path)

        attempt = 0
        while attempt < self.max_retries:
            command = self._build_ytdlp_command(ytid, file_path, start_seconds, end_seconds)
            os.system(command)

            duration = self.get_audio_duration(file_path) if os.path.exists(file_path) else None
            if duration and duration > 0.0:
                break  # Success

            attempt += 1
            print(f"[RETRY] Attempt {attempt}/{self.max_retries} failed for {file_path}. Retrying in {self.retry_delay}s...")

            # Refresh cookies if using browser cookies
            if self.cookies_from_browser:
                print("[INFO] Refreshing cookies from browser...")
                os.system('yt-dlp --cookies-from-browser firefox --skip-download https://www.youtube.com')

            time.sleep(self.retry_delay)

        if attempt == self.max_retries:
            print(f"[FAILED] Could not download file after {self.max_retries} attempts: {file_path}")
            return

        if self.copy_and_replicate:
            for label in positive_labels.split(',')[1:]:
                display_label = self.machine_to_display_mapping[label]
                target_path = os.path.join(self.root_path, display_label, f"{ytid}_{start_seconds}-{end_seconds}.{ext}")
                os.system(f'cp "{file_path}" "{target_path}"')
        return
