import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config
import engine
import dataset
from typing import List
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from model import HFAutoModel
from metrics.metrics import Metrics
from predict import load_model_weights, read_csv_sort_by_index, save_predictions_to_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-number', type=int, default=0)
    parser.add_argument('--best-model-dir', type=str, required=True)
    parser.add_argument('--train-data-csv', type=str, required=True)
    parser.add_argument('--valid-data-csv', type=str, required=True)
    parser.add_argument('--test-data-csv', type=str, required=True)
    parser.add_argument('--submission-file-path', type=str, default="submission.json")
    parser.add_argument('--train-data-batch-size', type=int, default=4)
    parser.add_argument('--valid-data-batch-size', type=int, default=4)
    args, _ = parser.parse_known_args()
    return args

def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True


def get_device(cuda_number):
    device:torch.device = torch.device(f'cuda:{cuda_number}') if torch.cuda.is_available() else torch.device('cpu')
    return device

def get_meta_classifier_inputs(train_outputs_all_models,
                                valid_outputs_all_models,
                                test_outputs_all_models,
                                train_targets,
                                valid_targets):
    test_train_out = np.array(train_outputs_all_models)
    test_valid_out = np.array(valid_outputs_all_models)
    test_test_out = np.array(test_outputs_all_models)

    X_train_meta = pd.DataFrame([test_train_out[:, i, :].reshape(-1)
                            for i in range(test_train_out.shape[1])])
    X_valid_meta = pd.DataFrame([test_valid_out[:, i, :].reshape(-1)
                                for i in range(test_valid_out.shape[1])])
    X_test_meta = pd.DataFrame([test_test_out[:, i, :].reshape(-1) 
                            for i in range(test_test_out.shape[1])])
    y_train_meta = train_targets
    y_valid_meta = valid_targets

    return X_train_meta, y_train_meta, X_valid_meta, y_valid_meta, X_test_meta



def get_predictions(data_loader, model, device):
    model.eval()
    fin_outputs = []
    with torch.no_grad():
        print("\n### GENERATING PREDICTIONS ###")
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            ids = torch.squeeze(ids).to(device, dtype=torch.long)
            token_type_ids = torch.squeeze(token_type_ids).to(device, dtype=torch.long)
            mask = torch.squeeze(mask).to(device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            m = nn.Softmax(dim=1)
            fin_outputs.extend(m(outputs).cpu().detach().numpy().tolist())
    return fin_outputs

def concat_prediction_get_meta_classifier_inputs(BEST_MODEL_DIR,
                                                train_data_loader, 
                                                valid_data_loader,
                                                test_data_loader, 
                                                device):
    train_outputs_all_models = []
    valid_outputs_all_models = []
    test_outputs_all_models = []
    train_targets = None
    valid_targets = None

    for model_file_name in os.listdir(BEST_MODEL_DIR):
        model = HFAutoModel()
        model_path = BEST_MODEL_DIR + f"/{model_file_name}"
        model = load_model_weights(model_path, model)
        model.to(device)
        
        outputs: List[List[float]]
        targets: List[float]
        valid_loss: float
        train_outputs, train_targets, train_loss = engine.eval_fn(train_data_loader, model, device)
        valid_outputs, valid_targets, valid_loss = engine.eval_fn(valid_data_loader, model, device)
        test_outputs = get_predictions(test_data_loader, model, device)
        train_outputs_all_models.append(train_outputs)
        valid_outputs_all_models.append(valid_outputs)
        test_outputs_all_models.append(test_outputs)
        
        print("\n### METRICS")
        print(f"Model Name: {model_file_name}")
        print(f"Train Loss: {train_loss}")
        print(f"Valid Loss: {valid_loss}")
    
    X_train_meta, y_train_meta, X_valid_meta, y_valid_meta, X_test_meta = get_meta_classifier_inputs(train_outputs_all_models,
                                                                                                    valid_outputs_all_models,
                                                                                                    test_outputs_all_models,
                                                                                                    train_targets,
                                                                                                    valid_targets)

    return X_train_meta, y_train_meta, X_valid_meta, y_valid_meta, X_test_meta
    

def train_meta_classifier(X_train, y_train):
    meta_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)
    meta_clf.fit(X_train, y_train)

    return meta_clf

def get_macro_f1_score(clf, X, y=None):
    pred = clf.predict(X)
    if y:
        f1_score_out = f1_score(y, pred, average='macro')
        return f1_score_out
    else:
        return pred
    
def run(seed, cuda_number, best_model_dir, train_data_csv, valid_data_csv,
        test_data_csv, train_data_batch_size, valid_data_batch_size,
        submission_file_path):
    seed_everything(seed)

    device = get_device(cuda_number)
    
    df_train = pd.read_csv(train_data_csv)
    df_valid = pd.read_csv(valid_data_csv)
    df_test = read_csv_sort_by_index(test_data_csv)
    index: list = df_test['index'].to_list()

    train_dataset = dataset.HFDataset(
    tweet=df_train.tweet.values,
    label=df_train.label.values
    )

    valid_dataset = dataset.HFDataset(
    tweet=df_valid.tweet.values,
    label=df_valid.label.values
    )

    test_dataset = dataset.HFDataset(
    tweet=df_test.tweet.values
    )

    train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_data_batch_size, num_workers=4
    )
    

    valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=valid_data_batch_size, num_workers=1
    )

    test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    X_train_meta, y_train_meta, X_valid_meta, y_valid_meta, X_test_meta = concat_prediction_get_meta_classifier_inputs(best_model_dir,
                                                                                                                    train_data_loader,
                                                                                                                    valid_data_loader,
                                                                                                                    test_data_loader,
                                                                                                                    device)


    meta_clf = train_meta_classifier(X_train_meta, y_train_meta)  


    train_f1_score = get_macro_f1_score(meta_clf, X_train_meta, y_train_meta)
    valid_f1_score = get_macro_f1_score(meta_clf, X_valid_meta, y_valid_meta)
    test_preds = get_macro_f1_score(meta_clf, X_test_meta)
    
    print("\n### F1-Score Metrics")
    print(f"Train F1 Score: {train_f1_score}")
    print(f"Valid Loss: {valid_f1_score}")

    print("\n Generating Submissions File")
    save_predictions_to_json(index, test_preds, submission_file_path)


if __name__ == "__main__":
    args = parse_args()
    SEED = args.seed
    BEST_MODEL_DIR = args.best_model_dir
    TRAIN_DATA_PATH = args.train_data_csv
    VALID_DATA_PATH = args.valid_data_csv
    TEST_DATA_PATH = args.test_data_csv
    SUBMISSION_FILE_PATH = args.submission_file_path
    TRAIN_DATA_BATCH_SIZE = args.train_data_batch_size
    VALID_DATA_BATCH_SIZE = args.valid_data_batch_size
    GPU_NUMBER = args.gpu_number
    DEVICE = get_device(GPU_NUMBER)

    run(
        SEED,
        GPU_NUMBER,
        BEST_MODEL_DIR,
        TRAIN_DATA_PATH,
        VALID_DATA_PATH,
        TEST_DATA_PATH,
        TRAIN_DATA_BATCH_SIZE,
        VALID_DATA_BATCH_SIZE,
        SUBMISSION_FILE_PATH
    )
