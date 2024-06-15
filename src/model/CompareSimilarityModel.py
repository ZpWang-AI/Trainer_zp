import numpy as np
import torch 
import torch.nn as nn 
import transformers

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          PreTrainedModel,
                          GenerationConfig,
                          )

from model import CustomModel, BaselineClassificationModel
from utils_zp.attr_dic import AttrDict


class CompareSimilarityConfig(AttrDict):
    def __init__(
        self, 
        base_model_path=None,
        num_labels=4,
        loss_type='CELoss',
    ) -> None:
        self.base_model_path = base_model_path
        self.num_labels = num_labels
        self.loss_type = loss_type
        
        
class CompareSimilarityModel(CustomModel):
    # predict_with_generate=False
    
    def __init__(
        self, 
        base_model_path,
        ft_model_path,
        num_labels,
        loss_type,
    ) -> None:        
        super().__init__()
        
        self.base_model_path = base_model_path
        self.ft_model_path = ft_model_path
        self.num_labels = num_labels
        self.loss_type = loss_type
        
        self.cls_model:BaselineClassificationModel = None
        self.encoder:nn.Module = None
        self.initial_model()
        
        self.cos_sim = nn.CosineSimilarity(dim=1)
    
    def initial_model(self):
        self.cls_model = BaselineClassificationModel(
            base_model_path=self.base_model_path,
            num_labels=self.num_labels,
            loss_type=self.loss_type,
        )
        self.cls_model.load_state_dict(torch.load(self.ft_model_path))
        self.encoder = self.cls_model.model.roberta
    
    def forward(self, tokenized_a, tokenized_b):
        za = self.encoder(**tokenized_a).last_hidden_state[:,0]
        zb = self.encoder(**tokenized_b).last_hidden_state[:,0]
        sim = self.cos_sim(za, zb)
        return sim
    