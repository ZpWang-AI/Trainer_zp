import numpy as np
import torch 
import torch.nn as nn 
import transformers

from collections import Counter
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
from utils_zp.attr_dic import AttrDict
from utils_zp import format_element_to_shape


class SubtextGenerationConfig(AttrDict):
    def __init__(
        self, 
        base_model_path=None,
        generation_config:dict={},
        label_list=None,
        # loss_type='CELoss',
    ) -> None:
        self.base_model_path = base_model_path
        self.generation_config = generation_config
        self.label_list = label_list
        # self.loss_type = loss_type
        
        
class SubtextGenerationModel(CustomModel):
    predict_with_generate=True
    
    def __init__(
        self, 
        base_model_path,
        generation_config,
        label_list,
        # loss_type,
    ) -> None:
        super().__init__()
        
        self.base_model_path = base_model_path
        self.label_list = label_list
        self.num_labels = len(label_list)
        
        self.model:PreTrainedModel = None
        self.model_config = None
        self.tokenizer = None
        self.initial_model()
        
        self.generation_config = GenerationConfig(**generation_config)
        self.vote_num = self.generation_config.num_return_sequences
    
    def initial_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_path, 
        )
        self.model_config = AutoConfig.from_pretrained(
            self.base_model_path, 
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
    
    def forward(self, input_ids, attention_mask, labels, *args, **kwargs):
        # if self.training:
        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # print(input_ids.shape, labels.shape)
        # outs = format_output(dict(model_outputs))
        # print(outs)
        # exit()
        logits = model_outputs.logits
        # pred = torch.softmax(logits, dim=-1)
        # loss = self.loss_fn(pred, labels)
        # generate_outputs = self.model.generate(input_ids, generation_config=self.generation_config)
        
        output = {
            'logits': logits,
            # 'pred': pred,
            'gt': labels,
            'loss': model_outputs['loss']
        }      
        return output
    
    def generate(self, input_ids, attention_mask, labels, *args, **kwargs):
        model_outputs = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            generation_config=self.generation_config
        )
        # pred = self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
        pred = model_outputs
        return pred
        