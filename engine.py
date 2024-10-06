import torch
import torch.nn as nn
from tqdm import tqdm

def loss_fn(outputs, targets, device):
    """
    This criteron computes the cross entropy loss between input logits and target.
    """
    weight = torch.tensor([0.8358099489795918, 0.9501903208265362,0.9534739905420153, 1.0294972505891595, 1.3680062630480168]).to(device)
    loss = nn.CrossEntropyLoss(weight)
    return loss(outputs, targets.long())

def train_fn(data_loader, model, optimizer, device, scheduler):
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

        loss = loss_fn(outputs, targets, device)
        
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
            loss = loss_fn(outputs, targets, device)
            
            total_valid_loss += loss.item()
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            m = nn.Softmax(dim=1)
            fin_outputs.extend(m(outputs).cpu().detach().numpy().tolist())
    avg_valid_loss = total_valid_loss / len(data_loader)
    return fin_outputs, fin_targets, avg_valid_loss




