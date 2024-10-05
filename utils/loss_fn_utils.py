import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import config
from enums.loss_enums import LossFuncEnum

def get_loss_function_weights(y_train, device):
    """
    Returns weights to assign to categorical crossentropy loss function to handle class imbalance problems.
    """
    if config.LOSS_FUNCTION == LossFuncEnum.WEIGHTED_CROSSENTROPY.value:
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train),
                                       y=y_train)
        return torch.tensor(weights, dtype=torch.float32).to(device)

    return None


class LossFunction:
    
    weight = [2.0615, 2.5864, 7.7958]
    def get_loss_func(loss_func: str):
        
        if loss_func == LossFuncEnum.BINARY_CROSSENTROPY.value:
            return nn.BCELoss()
    
        if loss_func == LossFuncEnum.CATEGORICAL_CROSSENTROPY.value:
            return nn.CrossEntropyLoss()
        
        if loss_func == LossFuncEnum.WEIGHTED_CROSSENTROPY.value:
            return nn.CrossEntropyLoss(weight=LossFunction.weight)
        
        if loss_func == LossFuncEnum.FOCAL_CROSSENTROPY.value:
            return FocalLoss()
        

class FocalLoss(nn.Module):
    
    def __init__(self, gamma=2.0, alpha=None, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma =gamma
        self.alpha =alpha
        self.reduction =reduction
        
    
    def get_aggregated_loss(self, loss):
        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'sum':
            return torch.sum(loss)
        if self.reduction == 'none':
            return loss
        
    def forward(self, inputs, targets):
        
        device = targets.device
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=torch.tensor(self.alpha, device=device))

        probs = torch.exp(-ce_loss)
        focal_loss = ((1 - probs) ** self.gamma) * ce_loss

        return self.get_aggregated_loss(focal_loss)
        
        
