import numpy as np
import torch 
import torch.nn as nn 
import transformers

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification,
                          PreTrainedModel,
                          T5Model,
                          GenerationConfig
                          )

from model.criterion import CELoss
from model import CustomModel
from utils.attr_dic import AttrDict
from utils import format_element_to_shape


class MultitaskConfig(AttrDict):
    def __init__(
        self, 
        base_model_path=None,
        num_labels=4,
        loss_type='CELoss',
        auxiliary_loss_weight=1.0,
    ) -> None:
        self.base_model_path = base_model_path
        self.num_labels = num_labels
        self.loss_type = loss_type
        self.auxiliary_loss_weight = auxiliary_loss_weight
        
        
class MultitaskModel(CustomModel):
    predict_with_generate=False
    
    def __init__(
        self, 
        base_model_path,
        num_labels,
        loss_type,
        auxiliary_loss_weight,
    ) -> None:
        super().__init__()
        
        self.base_model_path = base_model_path
        self.num_labels = num_labels
        self.auxiliary_loss_weight = auxiliary_loss_weight
        
        self.model_primary:nn.Module = None
        self.model_secondary:nn.Module = None
        self.model_config = None
        self.initial_model()
        # self.generation_config = GenerationConfig()
        
        if loss_type.lower() == 'celoss':
            self.loss_fn = CELoss()
        else:
            raise Exception('wrong loss_type')
    
    def initial_model(self):
        self.model_primary = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_path, 
            num_labels=self.num_labels,
        )
        self.model_secondary = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_path,
        )
        assert 't5' in self.base_model_path
        self.model_secondary.encoder = self.model_primary.transformer.encoder
        self.model_config = AutoConfig.from_pretrained(
            self.base_model_path, 
        )
        
        # self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
    
    def forward(self, input_ids, attention_mask, labels):
        # print(format_element_to_shape(input_ids))
        # print(format_element_to_shape(attention_mask))
        # print(format_element_to_shape(labels))
        # print('='*20)
        # exit()
        model_outputs = self.model_primary(input_ids=input_ids, attention_mask=attention_mask)
        logits = model_outputs.logits
        pred = torch.softmax(logits, dim=-1)
        
        if isinstance(labels, dict):
            loss = self.loss_fn(pred, labels['num'])
            
            model_outputs_auxiliary = self.model_secondary(input_ids=input_ids, attention_mask=attention_mask, labels=labels['str'])
            loss += model_outputs_auxiliary['loss']*self.auxiliary_loss_weight
        
        else:
            loss = self.loss_fn(pred, labels)
            
        return {
            'logits': logits,
            # 'pred': pred,
            'gt': labels,
            'loss': loss,
        }        

    def generate(self, input_ids, attention_mask, labels, *args, **kwargs):
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        pred = torch.argmax(outputs['logits'], dim=1)
        pred = torch.eye(self.num_labels, dtype=torch.long, device=input_ids.device)[pred]
        return pred
    