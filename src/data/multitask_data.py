import json
import os
import numpy as np
import pandas as pd
import torch

from typing import *
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score
    
from IDRR_data import DataFrames, PromptFiller
from data import CustomDataCollator, CustomDataset, CustomData, CustomComputeMetrics

from utils import format_element_to_shape


# class MultitaskComputeMetrics(CustomComputeMetrics):
#     def __init__(self, label_list:list) -> None:
#         self.label_list = label_list
#         self.metric_names = ['Macro-F1', 'Acc']+label_list
#         self.metric_names = ['Macro-F1']
        
#     def process_pred(self, pred):
#         pred = np.argmax(pred[0], axis=1)
#         pred = np.eye(len(self.label_list), dtype=int)[pred]
#         return pred


class MultitaskDataCollator(CustomDataCollator):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator_padding = DataCollatorWithPadding(self.tokenizer)
        # self.data_collator_seq2seq = DataCollatorForSeq2Seq(self.tokenizer)
        pass
        
    def __call__(self, features):
        if isinstance(features[0]['labels'], dict):
            y_strs = [p['labels']['str']for p in features]
            for p in features:
                p['labels'] = p['labels']['num']
            batch = self.data_collator_padding(features)
            max_str_len = max(len(p)for p in y_strs)
            for p in y_strs:
                p.extend([-100]*(max_str_len-len(p)))
            batch['labels'] = {
                'str': torch.tensor(y_strs, dtype=torch.long),
                'num': batch['labels']
            }
            pass
        
        else:
            batch = self.data_collator_padding(features)
        
        return batch
        print(batch)
        exit()
        # print(features)
        # exit()
        pass


# class MultitaskDataset(CustomDataset):
#     def __init__(
#         self, 
#         tokenizer,
#         x_strs:List[str], 
#         y_strs:List[str]=None,
#         y_nums:List[Union[float, List[float]]]=None,
#         # **kwargs
#     ) -> None:
#         super().__init__()
        
#         self.tokenizer = tokenizer
#         self.x_strs = x_strs
#         self.y_strs = y_strs
#         self.y_nums = y_nums
        
#     def __getitem__(self, index):
#         # print(self.x_strs[index])
#         model_inputs = self.tokenizer(
#             self.x_strs[index],
#             add_special_tokens=True, 
#             padding=True,
#             truncation='longest_first', 
#             max_length=1024,
#         )
        
#         if self.y_strs:
#             model_inputs['labels'] = {
#                 'str': self.tokenizer(
#                     self.y_strs[index],
#                     add_special_tokens=True, 
#                     padding=True,
#                     truncation='longest_first', 
#                     max_length=1024,
#                 ).input_ids,
#                 'num': self.y_nums[index],
#             }
        
#         else:
#             model_inputs['labels'] = self.y_nums[index]
    
#         return model_inputs
    
#     def __len__(self):
#         return len(self.x_strs)


class MultitaskData(CustomData):
    train_input_y_nums = True
    train_input_y_strs = True
    
    def get_data_collator(self):
        return MultitaskDataCollator(self.tokenizer)
    
    # def get_compute_metrics(self):
    #     return MultitaskComputeMetrics(
    #         label_list=self.label_list,
    #     )
    