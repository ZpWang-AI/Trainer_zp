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


class PCPConfig(AttrDict):
    def __init__(
        self, 
        base_model_path=None,
        label_list:list=None,
        ans_word_list_token:List[int]=None,
        ans_label_list_id:List[int]=None,
        loss_type='CELoss',
    ) -> None:
        self.base_model_path = base_model_path
        self.label_list = label_list
        self.ans_word_list_token = ans_word_list_token
        self.ans_label_list_id = ans_label_list_id
        self.loss_type = loss_type
        
        
class PCPModel(CustomModel):
    def __init__(
        self, 
        base_model_path,
        label_list,
        ans_word_list_token:List[int],
        ans_label_list_id:List[int],
        loss_type,
    ) -> None:
        super().__init__()
        
        self.base_model_path = base_model_path
        self.num_labels = len(label_list)
        
        self.model:nn.Module = None
        self.model_config = None
        self.tokenizer:transformers.PreTrainedTokenizer = None
        self.initial_model()
        self.generation_config = GenerationConfig()
        
        self.ans_word_list_token = ans_word_list_token
        self.ans_label_list_id = ans_label_list_id
        
        if loss_type.lower() == 'celoss':
            self.loss_fn = CELoss()
        else:
            raise Exception('wrong loss_type')
    
    def initial_model(self):
        self.model = transformers.RobertaForMaskedLM.from_pretrained(
            self.base_model_path, 
        )
        self.model_config = AutoConfig.from_pretrained(
            self.base_model_path, 
        )
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(
            self.base_model_path,
        )
    
    def forward(self, input_ids:torch.Tensor, attention_mask, labels):
        if labels.shape[1] == self.num_labels:
            return {'loss': torch.tensor(-1.0)}
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = model_outputs.logits  # bs, seq_len, vocab_size
        
        mask_positions = torch.where(input_ids == self.tokenizer.mask_token_id)
        mask_logits = logits[mask_positions]  # bs, vocab_size
        ans_word_logits = mask_logits[:, self.ans_word_list_token]  # bs, ans_word_num
        
        ans_word_prob = torch.softmax(ans_word_logits, dim=-1)  # bs
        loss = self.loss_fn(ans_word_prob, labels)
        return {
            'logits': ans_word_logits,
            'gt': labels,
            'loss': loss,
        }
    
    def generate(self, input_ids, attention_mask, labels, *args, **kwargs):
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = model_outputs.logits  # bs, seq_len, vocab_size
        
        mask_positions = torch.where(input_ids == self.tokenizer.mask_token_id)
        mask_logits = logits[mask_positions]  # bs, vocab_size
        ans_word_logits = mask_logits[:, self.ans_word_list_token]  # bs, ans_word_num
        
        ans_word_pred = torch.argmax(ans_word_logits, dim=-1)  # bs
        ans_label_pred = [self.ans_label_list_id[p]for p in ans_word_pred]
        pred_vec = torch.eye(self.num_labels, dtype=torch.long, 
                             device=input_ids.device)[ans_label_pred]
        return pred_vec
        