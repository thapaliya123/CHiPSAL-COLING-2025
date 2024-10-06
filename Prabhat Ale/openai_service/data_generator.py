import os
import numpy as np
import pandas as pd
from config import base_path, raw_train_csv_path, synthetic_data_path
from enums import SheetNames

# Load dataframes using the enum for sheet names
nepali_train_tweet_df = pd.read_excel(
    synthetic_data_path, sheet_name=SheetNames.NEPALI_TRAIN.value
)
hindi_train_tweet_df = pd.read_excel(
    synthetic_data_path, sheet_name=SheetNames.HINDI_TRAIN.value
)
nepali_valid_tweet_df = pd.read_excel(
    synthetic_data_path, sheet_name=SheetNames.NEPALI_VALID.value
)
hindi_valid_tweet_df = pd.read_excel(
    synthetic_data_path, sheet_name=SheetNames.HINDI_VALID.value
)
train_df = pd.read_csv(raw_train_csv_path)


train_synthetic_tweet_df = pd.concat([nepali_train_tweet_df, hindi_train_tweet_df])


# Create the index
index = np.arange(90001, 90001 + train_synthetic_tweet_df.shape[0])
# If you want to assign this index to the DataFrame
train_synthetic_tweet_df["index"] = index
train_synthetic_tweet_df["label"] = 2

final_train_synthetic_tweet_df = pd.DataFrame(
    {
        "index": train_synthetic_tweet_df["index"],
        "tweet": train_synthetic_tweet_df["Synthetic_Tweet"],
        "label": train_synthetic_tweet_df["label"],
    }
)

# Drop columns with any NaN values
final_train_synthetic_tweet_df = final_train_synthetic_tweet_df.dropna(
    how="any", axis=0
).reset_index(drop=True)

train_df = pd.concat([train_df, final_train_synthetic_tweet_df], axis=0)



print(f"The shape of nepali_train_tweet_df is {nepali_train_tweet_df.shape}")
print(f"The shape of nepali_valid_tweet_df is {nepali_valid_tweet_df.shape}")
print(f"The shape of hindi_train_tweet_df is {hindi_train_tweet_df.shape}")
print(f"The shape of hindi_valid_tweet_df is {hindi_valid_tweet_df.shape}")
print(f"The shape of train_synthetic_tweet_df is {train_synthetic_tweet_df.shape}")
print(f"The shape of train_df is {train_df.shape}")


train_df.to_csv(
    os.path.join(
        base_path,
        "data/synthetic_data/train_train/train_with_train_synthetic_combination.csv",
    ),
    index=False,
)