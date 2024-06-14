import json
import os
import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from typing import *
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score
    
from IDRR_data import PromptFiller
from data import CustomDataCollator, CustomDataset, CustomData, CustomComputeMetrics

from utils_zp import format_element_to_shape


# class GenerationComputeMetrics(CustomComputeMetrics):
#     def __init__(self, label_list:list, tokenizer, label_to_id_func) -> None:
#         self.label_list = label_list
#         self.tokenizer = tokenizer
#         self.label_to_id_func = label_to_id_func
#         self.vote_num = 1
        
#         self.metric_names = ['Macro-F1', 'Acc']+label_list
#         self.metric_names = ['Macro-F1']
    

    
# class GenerationDataCollator(CustomDataCollator):
#     pass


# class GenerationDataset(CustomDataset):
#     pass
#     # def __init__(
#     #     self, 
#     #     tokenizer,
#     #     x_strs:List[str], 
#     #     y_strs:List[str]=None,
#     #     # y_nums:List[Union[float, List[float]]]=None,
#     #     # **kwargs
#     # ) -> None:
#     #     super().__init__()
        
#     #     self.tokenizer = tokenizer
#     #     self.x_strs = x_strs
#     #     self.y_strs = y_strs
#     #     # self.y_nums = y_nums
        
#     # def __getitem__(self, index):
#     #     # print(self.x_strs[index])
#     #     model_inputs = self.tokenizer(
#     #         self.x_strs[index],
#     #         add_special_tokens=True, 
#     #         padding=True,
#     #         truncation='longest_first', 
#     #         max_length=1024,
#     #     )
#     #     model_inputs['labels'] = self.tokenizer(
#     #         self.y_strs[index],
#     #         add_special_tokens=True, 
#     #         padding=True,
#     #         truncation='longest_first', 
#     #         max_length=1024,
#     #     ).input_ids
#     #     # model_inputs['labels'] = self.y_nums[index]
    
#     #     return model_inputs
    
#     # def __len__(self):
#     #     return len(self.x_strs)


class GenerationData(CustomData):
    train_input_y_nums = False
    train_input_y_strs = True
    
    def get_data_collator(self):
        return DataCollatorForSeq2Seq(self.tokenizer)
    
    # def get_compute_metrics(self):
    #     return GenerationComputeMetrics(
    #         label_list=self.label_list,
    #         tokenizer=self.tokenizer,
    #         label_to_id_func=lambda x:self.dataframes.process_sense(sense=x,irrelevent_sense=self.num_labels)[1]
    #     )
    