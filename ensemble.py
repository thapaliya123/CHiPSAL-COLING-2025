import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List
from tqdm import tqdm
from scipy.stats import mode
from train import get_device, custom_metric
import config
from model import HFAutoModel
import dataset
from predict import load_model_weights

DEVICE = get_device(0)
MODEL = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--test-data-path', type=str, default='data/taskc/valid.csv')
    parser.add_argument('--test-batch-size', type=int, default=4)
    parser.add_argument('--soft-vote', action='store_true', help='Use Soft voting for ensemble predictions.')


    args, _ = parser.parse_known_args()
    return args

def load_model_weights(model, model_path):
    result = model.load_state_dict(torch.load(model_path))

    if len(result.missing_keys) > 0 or len(result.unexpected_keys) > 0:
        print("Warning: There are missing or unexpected keys.")
    else:
        print("Model loaded successfully.")
    
    return model

def read_csv_file(csv_file_path: str):
    df_test = pd.read_csv(csv_file_path)
    return df_test

def get_predictions(data_loader, model):
    model.eval()
    fin_outputs = []
    with torch.no_grad():
        print("\n### GENERATING PREDICTIONS ###")
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            ids = torch.squeeze(ids).to(DEVICE, dtype=torch.long)
            token_type_ids = torch.squeeze(token_type_ids).to(DEVICE, dtype=torch.long)
            mask = torch.squeeze(mask).to(DEVICE, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            m = nn.Softmax(dim=1)
            fin_outputs.extend(m(outputs).cpu().detach().numpy().tolist())
    return fin_outputs


def main(model_dir, test_data_path, test_batch_size, soft_vote=True):
    MODEL = HFAutoModel()
    df_test = read_csv_file(test_data_path)
    targets = df_test.label
    test_dataset = dataset.HFDataset(
        tweet=df_test.tweet.values
    )
    
    print(f"\n### Train Data Size: {len(test_dataset)} Rows")

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, num_workers=1
    )
    predictions = []
    for file_name in os.listdir(model_dir):
        print(f"Model Name: {file_name}")
        with torch.no_grad():
            MODEL = HFAutoModel()
            print(f"### Model: {file_name}")
            model_path = model_dir + f'/{file_name}'
            model = load_model_weights(MODEL, model_path)
            model.to(DEVICE)
            outputs: List[List[float]] = get_predictions(test_data_loader, model)
            predictions.append(outputs)
            # torch.cuda.empty_cache()
            # breakpoint()    
    final_prediction = np.mean(np.array(predictions), axis=0)
    df_ensemble_probs = pd.DataFrame(final_prediction) 
    if soft_vote:
        print("###SOFT VOTING###")
        ensemble_pred = np.argmax(final_prediction, axis=1)
    else:
        print("###HARD VOTING###")
        ensemble_pred = mode(np.argmax(predictions, axis=2), axis=0).mode
    metric_result = custom_metric.get_metrics_fn()(targets, ensemble_pred)
    print(f"Ensemble Metric result: {metric_result}")
    df_ensemble_probs.to_csv("./ensemble_results/test.csv", index=False)

if __name__ == '__main__':
    args = parse_args()
    MODEL_DIR = args.model_dir
    TEST_DATA_PATH = args.test_data_path
    TEST_BATCH_SIZE = args.test_batch_size
    SOFT_VOTE = args.soft_vote
    print(SOFT_VOTE)
    main(MODEL_DIR, TEST_DATA_PATH, TEST_BATCH_SIZE, soft_vote=SOFT_VOTE)