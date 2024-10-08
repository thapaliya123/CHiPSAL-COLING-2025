import config
import torch
import torch.nn as nn
from tqdm import tqdm
from enums.loss_enums import LossFuncEnum
from utils.loss_fn_utils import FocalLoss
# from train import get_loss_function_weights

focal_loss = FocalLoss()

def loss_fn(outputs, targets, weights=None):
    """
    This criteron computes the cross entropy loss between input logits and target.
    """
    if config.LOSS_FUNCTION in [LossFuncEnum.CATEGORICAL_CROSSENTROPY.value, LossFuncEnum.WEIGHTED_CROSSENTROPY.value]:
        loss = nn.CrossEntropyLoss(weight=weights)
        return loss(outputs, targets.long())
    else:
        return focal_loss(outputs, targets, alpha=weights)


def train_fn(data_loader, model, optimizer, device, scheduler, loss_fn_weights=None):
    model.train()
    total_train_loss = 0
    print("### Training Started ###")
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
        # print("Training!!")
        # print(f"ids: {ids}")
        # print(f"token_type_ids: {token_type_ids}")
        # print(f"mask: {mask}")
        # print(f"targets: {targets}")

        ids = torch.squeeze(ids).to(device, dtype=torch.long)
        token_type_ids = torch.squeeze(token_type_ids).to(device, dtype=torch.long)
        mask = torch.squeeze(mask).to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(outputs, targets, loss_fn_weights)
        
        total_train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_train_loss/len(data_loader)
    return avg_train_loss


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    total_valid_loss = 0
    print("### Evaluation Started ###")
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = torch.squeeze(ids).to(device, dtype=torch.long)
            token_type_ids = torch.squeeze(token_type_ids).to(device, dtype=torch.long)
            mask = torch.squeeze(mask).to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)


            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs, targets)
            
            total_valid_loss += loss.item()
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            m = nn.Softmax(dim=1)
            fin_outputs.extend(m(outputs).cpu().detach().numpy().tolist())
    avg_valid_loss = total_valid_loss / len(data_loader)
    return fin_outputs, fin_targets, avg_valid_loss




