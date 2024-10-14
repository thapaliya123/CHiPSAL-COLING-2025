import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dataset
from typing import List
from tqdm import tqdm
from scipy.stats import mode
from train import get_device
from model import HFAutoModel



DEVICE = None
MODEL = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--test-data-path', type=str, required=True)
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
        
def read_csv_from_path(csv_file_path: str):
    df = pd.read_csv(csv_file_path)
    return df

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
            outputs = torch.squeeze(outputs).to(DEVICE)
            m = nn.Softmax(dim=1)
            fin_outputs.extend(m(outputs).cpu().detach().numpy().tolist())
    return fin_outputs


def process_csv_get_predictions(test_data_path: str,
                                test_batch_size: int):
    
    
    df_test = read_csv_from_path(test_data_path)
    

    test_dataset = dataset.HFDataset(
        tweet=df_test.tweet.values
    )
    print(f"\n### Data Size: {len(test_dataset)} Rows")

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, num_workers=1, drop_last=True
    )

    outputs: List[List[float]] = get_predictions(test_data_loader, MODEL)
    outputs = np.argmax(np.array(outputs), axis=1)
    
    total_rows = len(df_test)
    n_dropped = total_rows % test_batch_size
    dropped_indices = list(range(total_rows - n_dropped, total_rows))
    df_test_dropped = df_test.drop(index=dropped_indices).reset_index(drop=True)
    targets = df_test_dropped.label[:len(outputs)]
    

    from sklearn.metrics import classification_report

    print(classification_report(targets, outputs, digits=5))
    mispredicted_indices = np.where(np.array(targets) != np.array(outputs))[0]
    df_mispredicted = df_test.iloc[mispredicted_indices]
    df_mispredicted['y_true'] = np.array(targets)[mispredicted_indices]
    df_mispredicted['y_pred'] = np.array(outputs)[mispredicted_indices]
    
    print("\n### MIS-PREDICTED EXAMPLES ###")
    print(df_mispredicted[['tweet', 'label', 'y_pred']])  # Display selected columns
    
    # Optionally, save the mispredicted data to a CSV for further analysis
    df_mispredicted.to_csv("./mis_predictions/mis_predictions_.csv", index=False)
    print("\n\nCompleted")


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
    # SUBMISSION_FILE_PATH = args.submission_file_path
    TEST_BATCH_SIZE = args.test_batch_size
    GPU_NUMBER = args.gpu_number
    DEVICE = get_device(GPU_NUMBER)
    print(DEVICE)
    load_model_weights(MODEL_PATH, MODEL)
    MODEL.to(DEVICE)
    MODEL.eval()
    process_csv_get_predictions(TEST_DATA_PATH, TEST_BATCH_SIZE)
    
    
# !python cls_report.py --model-path ./models/muril-large-cased-taska-f1_score-0.9973782634345743.bin --test-data-path ./data/taska/valid.csv 

# models/muril-large-cased-taskc-f1_score-categorical_crossentropy-freeze-layer-after-20-0.7747649186762761.bin

# python cls_report.py --model-path ./models/muril-large-cased-taskc-f1_score-0.7394279890552315-undersampling.bin --test-data-path ./data/taskc/valid.csv