import gdown
import pandas as pd

class DataIngestion:
    def __init__(self, file_id: list, save_filename: list):
        self.file_id: list = file_id
        self.filename: list = save_filename

    def download_drive_file_by_id(self, flag=False):
        if flag:
            for single_file_id, single_file_name in zip(self.file_id, self.filename):
                dataset_url = f"https://drive.google.com/u/1/uc?id={single_file_id}&export=download"
                gdown.download(dataset_url, single_file_name)


    def merge_valid_label_tweet_by_index(self, tweet_path, label_path, col_names):
        valid_df_tweet = pd.read_csv(tweet_path)
        valid_df_label = pd.read_csv(label_path)
        valid_df = pd.merge(valid_df_tweet, valid_df_label, on='index')
        valid_df.columns = COL_NAMES
        valid_df.drop('index', axis=1, inplace=True)
        return valid_df
    

    def save_dataframe_to_csv(self, df, save_path):
        df.to_csv(save_path, index=False)

if __name__ == "__main__":
    FILE_ID = ['166k7N9KV6jEDvvAr9iLTwrWyfdEcAs5p', '1-2TjS6xPfjWj9YaJGSf-JXXXfNz-2pNT', '1-1k1yHOGP7Wij1mUG2iKaSTN8i1WUgPz']
    SAVE_FILENAME = ["data/taskc/train.csv", "data/taskc/val_tweet.csv", "data/taskc/val_label.csv"]
    COL_NAMES = ['index', 'tweet', 'label']

    data_ingest = DataIngestion(FILE_ID, SAVE_FILENAME)
    data_ingest.download_drive_file_by_id()

    valid_df = data_ingest.merge_valid_label_tweet_by_index(SAVE_FILENAME[1], SAVE_FILENAME[2], COL_NAMES)
    data_ingest.save_dataframe_to_csv(valid_df, "data/taskc/valid.csv")



