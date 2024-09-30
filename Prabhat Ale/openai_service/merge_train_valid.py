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







synthetic_tweet_df = pd.concat([nepali_tweet_df, hindi_tweet_df])
# Create the index
index = np.arange(70001, 70001 + synthetic_tweet_df.shape[0])
# If you want to assign this index to the DataFrame
synthetic_tweet_df['index'] = index
synthetic_tweet_df['label'] = 2
final_synthetic_tweet_df = pd.DataFrame({'index': synthetic_tweet_df['index'],
                                         'tweet': synthetic_tweet_df['Synthetic_Tweet'],
                                         'label': synthetic_tweet_df['label']
                                         })
                                         
# Drop columns with any NaN values
final_synthetic_tweet_df = final_synthetic_tweet_df.dropna(how='any', axis=0)

train_df = pd.concat([train_df, final_synthetic_tweet_df], axis = 0)


train_valid_combined_df = pd.concat([train_df, valid_df], axis = 0)

train_valid_combined_df.to_csv(os.path.join(base_path, 'data/train_valid_synthetic_combination.csv'), index = False)