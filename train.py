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
from transformers import get_linear_schedule_with_warmup
from metrics.metrics import Metrics
from utils.wandb_utils import WandbLogger
from utils.loss_fn_utils import get_loss_function_weights
from enums import wandb_enums



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
    
    df_train = df_train.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    df_valid = df_valid.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    # df = pd.concat([df_train, df_valid])

    # TODO: ADD assert for df_train and df_valid columns

    train_dataset = dataset.HFDataset(
        tweet=df_train.tweet.values,
        label=df_train.label.values
    )

    valid_dataset = dataset.HFDataset(
        tweet=df_valid.tweet.values,
        label=df_valid.label.values
    )

    print(f"\n### Train Data Size: {len(train_dataset)} Rows")
    print(f"### Valid Data Size: {len(valid_dataset)} Rows\n")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4, shuffle=True
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1, shuffle=True
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
            print("### SAVING MODEL ###")
            best_model_path = config.MODEL_PATH+f"-{config.TASK_NAME}-{config.METRIC_NAME}-{config.LOSS_FUNCTION}-misclassifycorrected-{round(metric_result,6)}.bin"
            # best_model_path = "best_model_reproduce.bin"
            
            torch.save(model.state_dict(), best_model_path)
            best_metric_result = metric_result

        print("### WANDB LOGGING METRICS ###")
        wandb_logger.log_metrics({"train_loss": train_loss,
                                  "valid_loss": valid_loss,
                                  f"{config.METRIC_NAME}": metric_result
                                  })
    print("### WANDB LOGGING BEST MODEL ARTIFACT ###")
    wandb_logger.save_model_artifact(best_model_path)
    wandb_logger.finish()

if __name__ == "__main__":
    args = parse_args()
    cuda_number = args.gpu_number
    run(cuda_number)
