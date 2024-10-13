import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dataset
import config
from typing import List
from tqdm import tqdm
from scipy.stats import mode
from train import get_device
from model import HFAutoModel
from enums.ensemble_enums import EnsembleEnum, get_enum_values


DEVICE = None
MODEL = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--test-data-path', type=str, required=True)
    parser.add_argument('--submission-file-path', type=str, default="submission.json")
    parser.add_argument('--ensemble', type=str, choices=get_enum_values(EnsembleEnum), default=EnsembleEnum.NO_ENSEMBLE.value)
    parser.add_argument('--test-batch-size', type=int, default=4)
    parser.add_argument('--gpu-number', type=int, default=0)


    args, _ = parser.parse_known_args()
    return args

def load_model_weights(model_path, model = None):
    if not model:
        result = MODEL.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    else:
        result = model.load_state_dict(torch.load(model_path))
        return model
    
    if len(result.missing_keys) > 0 or len(result.unexpected_keys) > 0:
        print("Warning: There are missing or unexpected keys.")
    else:
        print("Model loaded successfully.")

def print_model_layers(model):
    print("\nMode Layers and Structures:\n")
    for name, layer in model.named_modules():
        print(f"Layer name: {name}\nLayer Structure: {layer}\n")

def read_csv_sort_by_index(csv_file_path: str):
    df_test = pd.read_csv(csv_file_path)
    assert list(df_test.columns) == ["index", "tweet"], "Invalid CSV file for predictions"
    df_test.sort_values(by='index', ascending=True, inplace=True)
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

def get_ensembled_predictions(model_dir, data_loader, ensemble):
    predictions = []
    for file_name in os.listdir(model_dir):
        print(f"Model Name: {file_name}")
        with torch.no_grad():
            MODEL = HFAutoModel()
            print(f"### Model: {file_name}")
            model_path = model_dir + f'/{file_name}'
            model = load_model_weights(model_path, MODEL)
            model.to(DEVICE)
            outputs: List[List[float]] = get_predictions(data_loader, model)
            predictions.append(outputs)
    final_prediction = np.mean(np.array(predictions), axis=0)

    if ensemble == EnsembleEnum.SOFT_VOTE.value:
        print("###SOFT VOTING###")
        ensemble_pred = np.argmax(final_prediction, axis=1)

    else:
        print("###HARD VOTING###")
        ensemble_pred = mode(np.argmax(predictions, axis=2), axis=0).mode
    return ensemble_pred

def save_predictions_to_json(index_list, predictions, json_file_path):
    assert json_file_path.split('/')[-1].split('.')[-1] == 'json', "Must pass file path with .json extension"
    predictions = [{"index": int(index_), "prediction": int(pred)}
                   for index_, pred in zip(index_list, predictions)]
    
    with open(json_file_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred)+'\n')

    print("### SUBMISSION FILE SAVE SUCCESSFULLY ###")

def process_csv_get_predictions(model_path: str,
                                test_data_path: str,
                                submission_file_path: str,
                                ensemble: str,
                                test_batch_size: int,
                                model = None):
    
    
    df_test = read_csv_sort_by_index(test_data_path)
    
    index: list = df_test['index'].to_list()

    test_dataset = dataset.HFDataset(
        tweet=df_test.tweet.values
    )
    print(f"\n### Train Data Size: {len(test_dataset)} Rows")

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, num_workers=1
    )

    if ensemble == EnsembleEnum.NO_ENSEMBLE.value:
        if not model:
            outputs: List[List[float]] = get_predictions(test_data_loader, MODEL)
        else:
            outputs: List[List[float]] = get_predictions(test_data_loader, model)
        outputs = np.argmax(np.array(outputs), axis=1)
    else:
        outputs = get_ensembled_predictions(model_path, test_data_loader, ensemble)

    print(outputs)
    save_predictions_to_json(index, outputs, submission_file_path)    


if __name__ == "__main__":
    TWEET = """
    सबै मिली सत्रे लाई हराऔँ🙏🙏
    #NoNotAgain 
    @cmprachanda
    """
    MODEL = HFAutoModel()
    args = parse_args()
    MODEL_PATH = args.model_path
    TEST_DATA_PATH = args.test_data_path
    SUBMISSION_FILE_PATH = args.submission_file_path
    TEST_BATCH_SIZE = args.test_batch_size
    GPU_NUMBER = args.gpu_number
    DEVICE = get_device(GPU_NUMBER)
    ENSEMBLE = args.ensemble
    print(DEVICE)
    try:
        load_model_weights(MODEL_PATH)
    except IsADirectoryError:
        print("MODEL IS DIRECTORY, Using ENSEMBLE")
    print_model_layers(MODEL)
    MODEL.to(DEVICE)
    MODEL.eval()
    process_csv_get_predictions(MODEL_PATH, TEST_DATA_PATH, SUBMISSION_FILE_PATH, ENSEMBLE, TEST_BATCH_SIZE)
    