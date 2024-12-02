from utils_zp import *
from utils_zp.ml import count_parameters

import numpy as np
import torch 
import torch.nn as nn 
import transformers

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          AutoModelForSeq2SeqLM,
                          GenerationConfig,
                          PreTrainedModel,
                          )

from .criterion import *
from .load_model import *


class CustomModelConfig(AttrDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.transformers_config = None
        self.base_model_name = None
        self.model_param_cnt = {
            'total': None,
            'trainable': None,
        }
    
    def refill(
        self,
        base_model_or_path,
        model:"CustomModel",
    ):
        self.base_model_name = path(base_model_or_path).stem
        self.model_param_cnt = count_parameters(model=model)


class CustomModel(nn.Module):
    def __init__(self, model_config:CustomModelConfig=None) -> None:
        super().__init__()
        self.model_config = model_config
        self.tokenizer = None
        self.generation_config = GenerationConfig()
    
    def initial_model(self):
        raise Exception()
    
    def forward(self): 
        raise Exception()
    
    def generate(self): 
        raise Exception('does not define generate func')