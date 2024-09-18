import os
import argparse
import pandas as pd
from tqdm import tqdm
from enum import Enum
from config import embedding_model_name
from embedding_model import EmbeddingModel
from config import (
    base_path, 
    valid_csv_path,
    train_csv_path,
    embedding_model_name
)


# Define an Enum for CSV path options
class CSVPathOption(Enum):
    TRAIN = 'train'
    VALID = 'valid'




if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description = "Generate embeddings from a cleaned tweet")
    parser.add_argument(
        "csv_option",
        type = CSVPathOption,
        choices = list(CSVPathOption),
        help = "Choose either 'train' for generating embeddings from training data or 'valid' for generating embeddings from validation data."
    )

    parser.add_argument(
        "-o", "--output_file_path",
        type = str, 
        default = "data/clean_train_df_emb_task3.csv",
        help = "Specify the path of the output file where data with embeddings need to be stored."
    )
    args = parser.parse_args()

    # Determine the CSV path based on the provided option
    if args.csv_option == CSVPathOption.TRAIN:
        input_file_path = train_csv_path
        print(train_csv_path)
    else:
        input_file_path = valid_csv_path
        print(valid_csv_path)
    embeddings = []
    df = pd.read_csv(input_file_path)
    emb_model = EmbeddingModel(embedding_model_name = embedding_model_name)
    # Iterate through each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        tweet = row['cleaned_tweet']  # Replace 'text' with the name of the column containing your text data
        embedding = emb_model.extract_embeddings(tweet)
        embeddings.append(embedding)

    # Optionally, add embeddings to the DataFrame
    df['tweet_embeddings'] = embeddings

    # Display the DataFrame with embeddings
    print(df.head())
    df.to_csv(os.path.join(base_path, args.output_file_path), index = False)
