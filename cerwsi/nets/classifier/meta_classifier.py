import torch
import torch.nn as nn
from abc import abstractmethod

class MetaClassifier(nn.Module):
    def __init__(self, evaluator, **agrs):
        super(MetaClassifier, self).__init__()
        self.evaluator = evaluator

    @property
    def device(self):
        return next(self.parameters()).device

    
    @abstractmethod
    def calc_logits(self, x: torch.Tensor):
        '''
        Args:
            x (tensor): input tensor
        Return:
            logits result (tuple)
        '''
    
    @abstractmethod
    def calc_loss(self, x, databatch):
        '''
        Args:
            x (tensor): input tensor
            databatch (dict): input with GT info
        Return:
            loss (float): loss can be back propagation
        '''
    
    @abstractmethod
    def set_pred(self, x, databatch):
        '''
        Args:
            x (tensor): input tensor
            databatch (dict): input with GT info
        Return:
            databatch (dict): update with Predict info
        '''
    
    