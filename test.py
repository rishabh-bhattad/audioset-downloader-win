import pandas as pd

df = pd.read_csv(
    f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_segments.csv", 
    sep=',', 
    skiprows=3,
    header=None,
    names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
    engine='python'
)

print("Total rows:", len(df))
print(df.head())