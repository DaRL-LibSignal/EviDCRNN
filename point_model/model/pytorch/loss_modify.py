import torch
import torch.nn as nn 

def masked_mae_loss(y_pred, y_true):
    loss = torch.abs(y_pred - y_true)
    loss[loss != loss] = 0
    return loss.mean()
def masked_mse_loss(y_pred, y_true):
    mse = nn.MSELoss()
    loss = mse(y_pred, y_true)
    loss[loss != loss] = 0
    return loss