import pandas as pd
import numpy as np
import os

# Paths
data_path = "data/train.csv"
output_dir = "data"

# Load dataset
df = pd.read_csv(data_path)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split index (20%)
split_index = int(0.8 * len(df))

# 80% (remaining training data)
train_data = df[:split_index]

# 20% (holdout data)
holdout_data = df[split_index:]

# OPTIONAL: overwrite or save new training file
train_data.to_csv(os.path.join(output_dir, "train_80.csv"), index=False)

# Split holdout into 4 equal parts
subsets = np.array_split(holdout_data, 4)

# Save subsets
for i, subset in enumerate(subsets, start=1):
    filename = f"input_{i}.csv"
    subset.to_csv(os.path.join(output_dir, filename), index=False)

print("✅ Done!")
print("Files created:")
print("- train_80.csv (80% data)")
for i in range(1, 5):
    print(f"- input_{i}.csv (5% each)")