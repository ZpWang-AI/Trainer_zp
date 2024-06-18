import numpy as np
import torch 
import torch.nn as nn 
import transformers

from typing import *
from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          GenerationConfig
                          )

from model import CustomModel
from model.criterion import CELoss
from utils_zp.attr_dic import AttrDict


class SubtextDiscriminatorConfig(AttrDict):
    def __init__(
        self, 
        base_model_path=None,
        loss_type='CELoss',
    ) -> None:
        self.base_model_path = base_model_path
        self.loss_type = loss_type
        
        
class SubtextDiscriminatorModel(CustomModel):
    def __init__(
        self, 
        base_model_path,
        loss_type,
    ) -> None:
        super().__init__()
        
        self.base_model_path = base_model_path
        
        self.model:nn.Module = None
        self.model_config = None
        self.tokenizer:transformers.PreTrainedTokenizer = None
        self.initial_model()
        self.generation_config = GenerationConfig()
        
        if loss_type.lower() == 'celoss':
            self.loss_fn = CELoss()
        else:
            raise Exception('wrong loss_type')
    
    def initial_model(self):
        self.model = transformers.RobertaForSequenceClassification.from_pretrained(
            self.base_model_path, num_labels=2,
        )
        self.model_config = AutoConfig.from_pretrained(
            self.base_model_path, 
        )
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(
            self.base_model_path,
        )
    
    def forward(self, input_ids:torch.Tensor, attention_mask, labels):
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = model_outputs.logits  # bs, seq_len, vocab_size
        
        pred = torch.softmax(logits, dim=-1)
        loss = self.loss_fn(pred, labels)
        return {
            'logits': logits,
            'gt': labels,
            'loss': loss,
        }
    
    def generate(self, input_ids, attention_mask, labels, *args, **kwargs):
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = model_outputs.logits  # bs, seq_len, vocab_size
        
        pred = torch.argmax(logits, dim=1)
        pred_vec = torch.eye(2, dtype=torch.long, device=input_ids.device)[pred]
        # print(pred_vec)
        return pred_vec
    
    def get_logits(self, input_ids, attention_mask):
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = model_outputs.logits  # bs, seq_len, vocab_size
        return logits