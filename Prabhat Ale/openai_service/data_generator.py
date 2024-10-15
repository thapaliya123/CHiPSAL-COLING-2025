import os
import numpy as np
import pandas as pd
from config import base_path, raw_train_csv_path, synthetic_data_path
from enums import SheetNames

# Load dataframes using the enum for sheet names
# nepali_train_tweet_df = pd.read_excel(
#     synthetic_data_path, sheet_name=SheetNames.NEPALI_TRAIN.value
# )
# hindi_train_tweet_df = pd.read_excel(
#     synthetic_data_path, sheet_name=SheetNames.HINDI_TRAIN.value
# )
nepali_valid_tweet_df = pd.read_excel(
    synthetic_data_path, sheet_name=SheetNames.NEPALI_VALID.value
)
hindi_valid_tweet_df = pd.read_excel(
    synthetic_data_path, sheet_name=SheetNames.HINDI_VALID.value
)
nepali_to_hindi_valid_tweet_df = pd.read_excel(
    synthetic_data_path, sheet_name=SheetNames.VALID_NEPALI_TO_HINDI.value
)
hindi_to_nepali_valid_tweet_df = pd.read_excel(
    synthetic_data_path, sheet_name=SheetNames.VALID_HINDI_TO_NEPALI.value
)
confusing_nepali_valid_tweet_df = pd.read_excel(
    synthetic_data_path, sheet_name=SheetNames.CONFUSING_NEPALI_TWEET.value
)
confusing_hindi_valid_tweet_df = pd.read_excel(
    synthetic_data_path, sheet_name=SheetNames.CONFUSING_HINDI_TWEET.value
)


train_df = pd.read_csv(raw_train_csv_path)


# train_synthetic_tweet_df = pd.concat([nepali_train_tweet_df, hindi_train_tweet_df], ignore_index=True)
valid_synthetic_tweet_df = pd.concat([nepali_valid_tweet_df, 
                                      hindi_to_nepali_valid_tweet_df,
                                      confusing_nepali_valid_tweet_df,
                                      hindi_valid_tweet_df, 
                                      nepali_to_hindi_valid_tweet_df,
                                      confusing_hindi_valid_tweet_df
                                      ], ignore_index = True)


# Create the index
# train_index = np.arange(90001, 90001 + train_synthetic_tweet_df.shape[0])
valid_index = np.arange(70001, 70001 + valid_synthetic_tweet_df.shape[0])

# If you want to assign this index to the DataFrame for training data
# train_synthetic_tweet_df["index"] = train_index
# train_synthetic_tweet_df["label"] = 2

# If you want to assign this index to the DataFrame for validation data
valid_synthetic_tweet_df["index"] = valid_index
valid_synthetic_tweet_df["label"] = 2

# final_train_synthetic_tweet_df = pd.DataFrame(
#     {
#         "index": train_synthetic_tweet_df["index"],
#         "tweet": train_synthetic_tweet_df["Synthetic_Tweet"],
#         "label": train_synthetic_tweet_df["label"],
#     }
# )


final_valid_synthetic_tweet_df = pd.DataFrame(
    {
        "index": valid_synthetic_tweet_df["index"],
        "tweet": valid_synthetic_tweet_df["Synthetic_Tweet"],
        "label": valid_synthetic_tweet_df["label"],
    }
)


# Drop columns with any NaN values for training data
# final_train_synthetic_tweet_df = final_train_synthetic_tweet_df.dropna(
#     how="any", axis=0
# ).reset_index(drop=True)

# train_with_synthetic_train_df = pd.concat([train_df, final_train_synthetic_tweet_df], axis=0)



# Drop columns with any NaN values for validation data
final_valid_synthetic_tweet_df = final_valid_synthetic_tweet_df.dropna(
    how="any", axis=0
).reset_index(drop=True)

train_with_synthetic_valid_df = pd.concat([train_df, final_valid_synthetic_tweet_df], axis=0)



# print(f"The shape of nepali_train_tweet_df is {nepali_train_tweet_df.shape}")
print(f"The shape of nepali_valid_tweet_df is {nepali_valid_tweet_df.shape}")
# print(f"The shape of hindi_train_tweet_df is {hindi_train_tweet_df.shape}")
print(f"The shape of hindi_valid_tweet_df is {hindi_valid_tweet_df.shape}")
# print(f"The shape of train_synthetic_tweet_df is {train_synthetic_tweet_df.shape}")
print(f"The shape of train_df is {train_df.shape}")


print(train_with_synthetic_valid_df.shape)
# train_with_synthetic_train_df.to_csv(
#     os.path.join(
#         base_path,
#         "data/synthetic_data/train_train/train_with_train_synthetic_combination.csv",
#     ),
#     index=False,
# )


train_with_synthetic_valid_df.to_csv(
    os.path.join(
        base_path,
        "data/synthetic_data/train_train/final_train_with_valid_synthetic_combination_v2.csv",
    ),
    index=False,
)