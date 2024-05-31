import json
import os
import numpy as np
import pandas as pd

from typing import *
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score
    
from IDRR_data import DataFrames, PromptFiller
from data import (CustomDataCollator, CustomDataset, 
                  CustomData, CustomComputeMetrics)


# class ClassificationComputeMetrics(CustomComputeMetrics):
#     # metric_names: List[str]
#     def __init__(self, label_list:list) -> None:
#         self.label_list = label_list
#         self.metric_names = ['Macro-F1', 'Acc']+label_list
#         self.metric_names = ['Macro-F1']
        
#     def process_pred(self, pred):
#         pred = np.argmax(pred[0], axis=1)
#         pred = np.eye(len(self.label_list), dtype=int)[pred]
#         return pred
        
    

# class ClassificationDataCollator(CustomDataCollator):
#     pass


# class ClassificationDataset(CustomDataset):
#     pass
#     def __init__(
#         self, 
#         tokenizer,
#         x_strs:List[str], 
#         y_nums:List[Union[float, List[float]]]=None,
#         # **kwargs
#     ) -> None:
#         super().__init__()
        
#         self.tokenizer = tokenizer
#         self.x_strs = x_strs
#         self.y_nums = y_nums
        
#     def __getitem__(self, index):
#         model_inputs = self.tokenizer(
#             self.x_strs[index],
#             add_special_tokens=True, 
#             padding=True,
#             truncation='longest_first', 
#             max_length=1024,
#         )
#         # model_inputs['labels'] = self.tokenizer(
#         #     self.y_strs[index],
#         #     add_special_tokens=True, 
#         #     padding=True,
#         #     truncation='longest_first', 
#         #     max_length=256,
#         # ).input_ids
#         model_inputs['labels'] = self.y_nums[index]
        
#         # for key in self.extra_kwargs:
#         #     model_inputs[key] = self.extra_kwargs[key][index]
    
#         return model_inputs
    
#     def __len__(self):
#         return len(self.x_strs)


class ClassificationData(CustomData):
    train_input_y_nums = True
    train_input_y_strs = False
    
    def get_data_collator(self):
        return DataCollatorWithPadding(self.tokenizer)
    
    # def get_compute_metrics(self):
    #     return CustomComputeMetrics(self.label_list)
        # return ClassificationComputeMetrics(self.label_list)
    