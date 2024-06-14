import json
import os
import numpy as np
import pandas as pd
import transformers

from collections import defaultdict, Counter
from typing import *
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score
from nltk.translate.meteor_score import meteor_score
    
from IDRR_data import PromptFiller
from data import CustomDataCollator, CustomDataset, CustomData, CustomComputeMetrics
from utils_zp import format_element_to_shape


class SubtextDistilComputeMetrics(CustomComputeMetrics):
    def __init__(self, label_list:list, tokenizer:transformers.PreTrainedTokenizer) -> None:
        self.label_list = label_list
        self.vote_num = 1
        self.tokenizer = tokenizer
        
        # self.metric_names = ['Macro-F1', 'Acc']+label_list
        self.metric_names = ['Meteor']
    
    def __call__(self, eval_pred):
        """
        pred: ndarray [datasize, seq_len]
        labels: ndarray [datasize, seq_len]
        """
        pred, labels = eval_pred
        pred = np.where(pred==-100, self.tokenizer.pad_token_id, pred)
        labels = np.where(labels==-100, self.tokenizer.pad_token_id, labels)
        pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        meteor_scores = []
        for cpred, clabel in zip(pred, labels):
            cpred = self.tokenizer.tokenize(cpred)
            clabel = self.tokenizer.tokenize(clabel)
            meteor_scores.append(meteor_score(
                references=[clabel], hypothesis=cpred,
            ))
        
        res = {
            'Meteor': np.mean(meteor_scores)
        }
        return res
        
    
# class GenerationDataCollator(CustomDataCollator):
#     pass


# class GenerationDistilDataset(CustomDataset):
#     def __init__(
#         self, 
#         tokenizer,
#         x_strs:List[str], 
#         y_strs:List[str]=None,
#         max_length:
#         # y_nums:List[Union[float, List[float]]]=None,
#         # **kwargs
#     ) -> None:
#         super().__init__()
        
#         self.tokenizer = tokenizer
#         self.x_strs = x_strs
#         self.y_strs = y_strs
#         # self.y_nums = y_nums
        
#     def __getitem__(self, index):
#         # print(self.x_strs[index])
#         model_inputs = self.tokenizer(
#             self.x_strs[index],
#             add_special_tokens=True, 
#             padding=True,
#             truncation='longest_first', 
#             max_length=1024,
#         )
#         model_inputs['labels'] = self.tokenizer(
#             self.y_strs[index],
#             add_special_tokens=True, 
#             padding=True,
#             truncation='longest_first', 
#             max_length=1024,
#         ).input_ids
#         # model_inputs['labels'] = self.y_nums[index]
    
#         return model_inputs
    
#     def __len__(self):
#         return len(self.x_strs)


class SubtextDistilData(CustomData):
    train_input_y_nums = False
    train_input_y_strs = True
    
    def get_data_collator(self):
        return DataCollatorForSeq2Seq(self.tokenizer)
    
    def get_compute_metrics(self):
        return SubtextDistilComputeMetrics(self.label_list, self.tokenizer)
    
    def get_dataset(
        self,
        split:Literal['train', 'dev', 'test'],
    ) -> CustomDataset:
        df = self.get_preprocessed_dataframe(split=split)
        df = df[df['subtext']!='']
        
        if split == 'train':
            return CustomDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(df=df, prompt=self.prompt['train_x'],
                                         tokenizer=self.tokenizer)),
                y_strs=list(PromptFiller(df=df, prompt=self.prompt['train_y'],
                                         tokenizer=self.tokenizer)),
                max_length=self.max_length,
                shift_num=self.shift_num,
            )
        else:
            return CustomDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(df=df, prompt=self.prompt['eval_x'], 
                                         tokenizer=self.tokenizer,
                                         ignore=self.test_x_ignore,
                                         )),
                y_strs=list(PromptFiller(df=df, prompt=self.prompt['eval_y'],
                                         tokenizer=self.tokenizer)),
                max_length=self.max_length,
            )
    