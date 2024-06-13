import numpy as np
import torch 
import torch.nn as nn 
import transformers

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          GenerationConfig
                          )

from model.criterion import CELoss
from utils_zp.attr_dic import AttrDict


class CustomModel(nn.Module):
    # predict_with_generate:bool
    
    def __init__(self) -> None:
        super().__init__()
        self.generation_config = GenerationConfig()
    
    def initial_model(self):
        raise Exception()
    
    def forward(self): 
        raise Exception()
    
    def generate(self): 
        raise Exception('does not define generate func')