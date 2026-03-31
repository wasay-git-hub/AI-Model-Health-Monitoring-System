import pandas as pd
import numpy as np
import os

# Paths
data_path = "data/train.csv"
output_dir = "data"

# Load dataset
df = pd.read_csv(data_path)

# Sort chronologically so the holdout is the most recent 20%
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

# Split index (80% train, 20% most recent holdout)
split_index = int(0.8 * len(df))

# 80% (remaining training data)
train_data = df[:split_index]

# 20% (most recent holdout data)
holdout_data = df[split_index:]

# overwrite or save new training file
train_data.to_csv(os.path.join(output_dir, "train_80.csv"), index=False)

# Split holdout into 4 equal parts without DataFrame swapaxes warnings
subset_indices = np.array_split(np.arange(len(holdout_data)), 4)
subsets = [holdout_data.iloc[idx].copy() for idx in subset_indices]

# Save subsets
for i, subset in enumerate(subsets, start=1):
    filename = f"input_{i}.csv"
    subset.to_csv(os.path.join(output_dir, filename), index=False)

print("Done!")
print("Files created:")
print("- train_80.csv (80% data)")
for i in range(1, 5):
    print(f"- input_{i}.csv (5% each)")