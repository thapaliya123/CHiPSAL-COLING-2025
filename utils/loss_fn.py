import torch
import torch.nn as nn
import torch.nn.functional as F
from enums.loss_enums import LossFuncEnum
import config
import pandas as pd
import numpy as np


class LossFunction:
    
    device = 'cuda'
    weight = torch.tensor([0.63175434, 2.39747064]).to(device)
    
    def get_loss_func(loss_func: str):
        
        if loss_func == LossFuncEnum.BINARY_CROSSENTROPY.value:
            return nn.BCELoss()
    
        if loss_func == LossFuncEnum.CATEGORICAL_CROSSENTROPY.value:
            return nn.CrossEntropyLoss()
        
        if loss_func == LossFuncEnum.WEIGHTED_CROSSENTROPY.value:
            return nn.CrossEntropyLoss(weight=LossFunction.weight)
        
        if loss_func == LossFuncEnum.FOCAL_CROSSENTROPY.value:
            return FocalLoss(alpha=LossFunction.weight)
        
        assert loss_func in [loss.value for loss in LossFuncEnum], 'Please check loss function'
        

class FocalLoss(nn.Module):
    
    def __init__(self, gamma=4.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma =gamma
        self.alpha =alpha
        self.reduction =reduction
        
    
    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # pt is the predicted probability for the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
        
