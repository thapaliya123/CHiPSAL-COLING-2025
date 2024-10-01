import os
import numpy as np
import pandas as pd
from config import (
    base_path,
    raw_train_csv_path,
    raw_valid_csv_path,
    raw_test_csv_path,
    synthetic_data_path,

)
from enums import SheetNames

# Load dataframes using the enum for sheet names
nepali_tweet_df = pd.read_excel(synthetic_data_path, sheet_name=SheetNames.NEPALI.value)
hindi_tweet_df = pd.read_excel(synthetic_data_path, sheet_name=SheetNames.HINDI.value)
train_df = pd.read_csv(raw_train_csv_path)
valid_df = pd.read_csv(raw_valid_csv_path)
test_df = pd.read_csv(raw_test_csv_path)

print(train_df.shape)

train_valid_combined_df = pd.concat([train_df, valid_df], axis = 0)
print(train_valid_combined_df.shape)



train_df.to_csv(os.path.join(base_path, 'data/raw_train_valid_combination.csv'), index = False)