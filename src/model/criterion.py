import torch
import torch.nn as nn


# make sure `y_true.dim = 2`
class CELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        """
        n = label categories
        y_pred: [batch size, n]
        y_true: [batch size, n]
        
        formula:
            y_pred = softmax(logits)
            loss = sum( y_true * log(y_pred) )
        """ 
        # y_pred = torch.softmax(logits, dim=1)
        return -(y_true*torch.log(y_pred)).sum(dim=1).mean()
    
    
class KLDivLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        """
        n = feature dimension
        y_pred: [batch size, n]
        y_true: [batch size, n]
        
        formula:
            y_pred = softmax(logits)
            loss = sum( y_true * (log(y_true)-log(y_pred)) )
        """ 
        # y_pred = torch.softmax(logits, dim=1)
        return ( y_true*(torch.log(y_true)-torch.log(y_pred)) ).sum(dim=1).mean()