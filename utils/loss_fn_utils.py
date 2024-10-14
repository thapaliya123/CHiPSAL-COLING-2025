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

class FocalLoss(nn.Module):
  def __init__(self, gamma=2.0, reduction='mean'):
    """
    Args:
      gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
      alpha (list, optional): Weights for each class. Can be used to tackle class imbalance._
      reduction (str, optional): Specifies the reduction to apply to the output. Can be 'mean', 'sum', or 'none'.
    """
    super(FocalLoss, self).__init__()
    self.gamma = gamma
    self.reduction = reduction

  def get_aggregated_loss(self, loss):
    if self.reduction == 'mean':
      return torch.mean(loss)
    elif self.reduction == 'sum':
      return torch.sum(loss)
    elif self.reduction == 'none':
      return loss

  def forward(self, inputs, targets, alpha=None):
    # Calculate cross-entropy loss
    ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=alpha)

    # Get probabilites from Crossentropy Loss
    probs = torch.exp(-ce_loss)

    # Compute Focal Loss
    focal_loss = ((1 - probs) ** self.gamma) * ce_loss
    # breakpoint()
    return self.get_aggregated_loss(focal_loss)