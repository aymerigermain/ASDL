# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implementation of the Focal Loss :math:`\\frac{1}{N} \\sum_i -(1-p_{y_i} + \\epsilon)^\\gamma \\log(p_{y_i})`

    Args:
        gamma: :math:`\\gamma > 0` puts more focus on hard misclassified samples

    Shape:
        - predictions :math:`(B, C)` : the logits
        - targets :math`(B, )` : the target ids to predict
    """

    def __init__(self, gamma=2.0, ignore_index: int = None):
        super().__init__()
        self.gamma = gamma
        self.eps = 1e-10
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """
        Arguments:
            predictions: (B, K, d1, d2, ..) of pre-activation
            targets : (B, d1, d2, ..) of class indices
        """
        B, num_classes = predictions.shape[:2]
        predictions = predictions.view(B, num_classes, -1)  # B, K, d1*d2*d3*..
        predictions = predictions.transpose(1, -1)  # B, d1*d2*d3*...., K
        predictions = predictions.reshape(-1, num_classes)

        targets = targets.view(-1)

        if self.ignore_index is not None:
            predictions = predictions[targets != self.ignore_index]
            targets = targets[targets != self.ignore_index]

        targets = targets.unsqueeze(dim=-1)
        logp = F.log_softmax(predictions, dim=1)
        logp_t = logp.gather(dim=1, index=targets)

        weight = torch.pow(1.0 - torch.exp(logp_t) + self.eps, self.gamma)
        loss = (-weight * logp_t).mean()

        return loss
