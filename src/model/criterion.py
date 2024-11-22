import torch
import torch.nn as nn


class CELoss(nn.Module):
    """
    n = label categories
    
    y_pred: [batch size, n]
    
    y_true: [batch size, n]
    
    math:
        < y_pred = softmax(logits) >
        loss = sum( - y_true * log(y_pred) ), mean on batch
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        # y_pred = torch.softmax(logits, dim=1)
        return -(y_true*torch.log(y_pred)).sum(dim=1).mean()
    
    
class KLDivLoss(nn.Module):
    """
    n = feature dimension
    
    y_pred: [batch size, n]
    
    y_true: [batch size, n]
    
    math:
        < y_pred = softmax(logits) >
        loss = sum( y_true * (log(y_true)-log(y_pred)) ), mean on batch
    """ 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        # y_pred = torch.softmax(logits, dim=1)
        return ( y_true*(torch.log(y_true)-torch.log(y_pred)) ).sum(dim=1).mean()


if __name__ == '__main__':
    gt = torch.tensor([
        [1,0,0],
        [0,1,0],
    ])
    pred = torch.tensor([
        [0.1,0.2,0.7],
        [0.2,0.3,0.5],
    ])
    print(-(gt*torch.log(pred)).sum(dim=1))
    print(CELoss()(pred, gt))
    print(nn.KLDivLoss(log_target=True)())