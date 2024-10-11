import argparse
from datetime import datetime
from typing import List
import config
import dataset
import engine
import torch
import random
import pandas as pd
import torch.nn as nn
import numpy as np

from model import HFAutoModel
from sklearn import model_selection, metrics
from sklearn.model_selection import StratifiedKFold
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from metrics.metrics import Metrics
from utils.wandb_utils import WandbLogger
from utils.loss_fn_utils import get_loss_function_weights
from enums import wandb_enums,ensemble_enums
from predict import load_model_weights, process_csv_get_predictions



WANDB_RUN_NAME = f"{config.HF_MODEL_PATH.split('/')[-1]}+TASK: {config.TASK_NAME} {str(datetime.now())}"
CONFIG_DICT = {
                wandb_enums.WandbEnum.WANDB_PROJECT_NAME.value: config.WANDB_PROJECT_NAME,
                wandb_enums.WandbEnum.WANDB_TAGS.value: config.TAGS,
                "seed": config.SEED,
                "epochs": config.EPOCHS,
                "train_batch_size": config.TRAIN_BATCH_SIZE,
                "valid_batch_size": config.VALID_BATCH_SIZE,
                "lr": config.LEARNING_RATE,
                "loss_function": config.LOSS_FUNCTION,
                "augmentation": config.DATA_AUGMENTATION,
                "preprocessing": config.PREPROCESSING
    }

custom_metric = Metrics(config.METRIC_NAME)
wandb_logger = WandbLogger(WANDB_RUN_NAME, CONFIG_DICT, 
                           config.LOG_TO_WANDB)
wandb_logger.initialize()

def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-number', type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

def seed_worker(worker_id):
    """
    seed each worker in DataLoader to ensure reproducibility.
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_device(cuda_number):
    device:torch.device = torch.device(f'cuda:{cuda_number}') if torch.cuda.is_available() else torch.device('cpu')
    return device

def get_optimizer_parameters(named_parameters: list):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in named_parameters if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in named_parameters if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_parameters

def run(cuda_number):
    # seed everything for reproducibility
    seed_everything(config.SEED)

    df_train = pd.read_csv(config.TRAINING_FILE).drop('index', axis=1)
    df_valid = pd.read_csv(config.VALID_FILE)
    df = pd.concat([df_train, df_valid])
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=config.SEED)

    # breakpoint()

    # TODO: ADD assert for df_train and df_valid columns

    df_dataset = dataset.HFDataset(
        tweet=df.tweet.values,
        label=df.label.values
    )

    # breakpoint()

    print(f"\n### Train + Valid Data Size: {len(df_dataset)}")


    fold_f1_score = []
    fold_best_models = []
    fold_best_models_path = []
    for fold, (train_indices, val_indices) in enumerate(skf.split(df['tweet'], df['label'])):
        # print(fold)
        # breakpoint()
        
        df_train = df.iloc[train_indices, :]
        df_valid = df.iloc[val_indices, :]

        train_dataset = dataset.HFDataset(
        tweet=df_train.tweet.values,
        label=df_train.label.values
        )

        valid_dataset = dataset.HFDataset(
            tweet=df_valid.tweet.values,
            label=df_valid.label.values
        )
        
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
        )

        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
        )
        
        device = get_device(cuda_number)

        model = HFAutoModel()
        # model.print_trainable_layers()
        model.to(device)

        param_optimizer = list(model.named_parameters())
        optimizer_parameters: List[dict] = get_optimizer_parameters(param_optimizer)
        num_train_steps = int((len(df_train) / config.TRAIN_BATCH_SIZE) * config.EPOCHS)
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )
        loss_fn_weights = get_loss_function_weights(df_train.label.values, device)
        
        best_metric_result = 0
        best_model = model
        for epoch in range(config.EPOCHS):
            print(f"\n##### EPOCH {epoch+1} #####")
            
            train_loss: float = engine.train_fn(train_data_loader, model, optimizer, device, scheduler,
                                                loss_fn_weights)
            
            outputs: List[List[float]]
            targets: List[float]
            valid_loss: float
            outputs, targets, valid_loss = engine.eval_fn(valid_data_loader, model, device)
            outputs = np.argmax(np.array(outputs), axis=1)
            
            metric_result = custom_metric.get_metrics_fn()(targets, outputs)
            print(f"\nTRAIN LOSS: {train_loss}")
            print(f"VALIDATION LOSS: {valid_loss}")
            print(f"METRIC (Validation Split): {config.METRIC_NAME} --> {metric_result}")
            if metric_result > best_metric_result:
                print(f"\nBest Metric: {metric_result}")
                # print("### SAVING MODEL ###")
                best_model_path = config.MODEL_PATH+f"-{config.TASK_NAME}-fold-{fold}-{config.METRIC_NAME}-{config.LOSS_FUNCTION}-{metric_result}.bin"
                # torch.save(model.state_dict(), best_model_path)
                best_metric_result = metric_result
                best_model = model

        fold_f1_score.append(best_metric_result)
        # fold_best_models.append(best_model)
        torch.save(best_model.state_dict(), best_model_path)
        fold_best_models_path.append(best_model_path)
        
    print("\n###Average Fold Metrics:")
    print(np.array(fold_f1_score).mean().item())
    print(fold_f1_score)

    ## Generate predictions on test samples
    TEST_DATA_PATH = "data/taskc/test.csv"
    ensemble = ensemble_enums.EnsembleEnum.NO_ENSEMBLE.value
    SUBMISSION_PATH = "./submissions"
    print(f'\n Generating Submissions for all folds.')
    # for fold_num, (best_model_path, model) in enumerate(zip(fold_best_models_path, fold_best_models)): 
    df_submission_list = []
    for fold_num, best_model_path in enumerate(fold_best_models_path): 
        print(f"fold: {fold_num} --> Saved Model")
        model = HFAutoModel()
        model = load_model_weights(best_model_path, model)
        submission_path = SUBMISSION_PATH+f"/taskc-fold-{fold_num}.json"
        process_csv_get_predictions(best_model_path,
                                    TEST_DATA_PATH,
                                    submission_path,
                                    ensemble,
                                    4,
                                    model)
        df = pd.read_json(submission_path, lines=True)
        df.set_index('index', inplace=True)
        df_submission_list.append(df)
    
    df_concat = pd.concat(df_submission_list, axis=1)
    df_concat = pd.DataFrame(df_concat.mode(axis=1)[0])
    df_concat.columns = ['prediction']
    df_concat.reset_index(inplace=True)
    df_concat = df_concat.astype({"index": int, "prediction": int})
    df_concat.to_json('submissions/fold_final.json', orient='records', lines=True)


    
if __name__ == "__main__":
    args = parse_args()
    cuda_number = args.gpu_number
    run(cuda_number)
