import os
import torch
import wandb
import gdown
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from datasets import Dataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from google.colab import userdata
from transformers import Trainer, TrainingArguments
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer,
                          TrainingArguments,
                          EarlyStoppingCallback
                          )
from torch.nn import CrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_drive_file_by_id(file_id, target_file_name):
  dataset_url = f"https://drive.google.com/u/1/uc?id={file_id}&export=download"
  gdown.download(dataset_url, target_file_name)
  print("File Download succesfull")

FILE_ID = ['166k7N9KV6jEDvvAr9iLTwrWyfdEcAs5p', '1-2TjS6xPfjWj9YaJGSf-JXXXfNz-2pNT', '1-1k1yHOGP7Wij1mUG2iKaSTN8i1WUgPz']
FILENAME = ["train.csv", "val_tweet.csv", "val_label.csv"]

for file_id, file_name in zip(FILE_ID, FILENAME):
    if not os.path.exists(file_name):
        download_drive_file_by_id(file_id, file_name)
    else:
        print(f"File '{file_name}' already exists.")

COL_NAMES = ['index', 'text', 'label']
train_df = pd.read_csv(FILENAME[0], header=0, names=COL_NAMES)
valid_df_tweet = pd.read_csv(FILENAME[1])
valid_df_label = pd.read_csv(FILENAME[2])
valid_df = pd.merge(valid_df_tweet, valid_df_label, on='index')
valid_df.columns = COL_NAMES
train_df.drop('index', axis=1, inplace=True)
valid_df.drop('index', axis=1, inplace=True)

def tokenize_function(examples):
  return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=169)

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

model_name = "xlm-roberta-base"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model