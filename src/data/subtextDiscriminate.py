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
    
from IDRR_data import DataFrames, PromptFiller
from data import CustomDataCollator, CustomDataset, CustomData, CustomComputeMetrics
from utils_zp import format_element_to_shape


class SubtextDiscriminateComputeMetrics(CustomComputeMetrics):
    def __init__(self, label_list:list, tokenizer:transformers.PreTrainedTokenizer) -> None:
        self.label_list = label_list
        self.vote_num = 1
        self.tokenizer = tokenizer
        
        # self.metric_names = ['Macro-F1', 'Acc']+label_list
        self.metric_names = ['F1']
    
    def __call__(self, eval_pred):
        """
        n = label categories
        eval_pred: (pred, labels)
        pred: np.array [datasize, n]
        labels: np.array [datasize, n]
        # X[p][q]=True, sample p belongs to label q (False otherwise)
        """
        pred, labels = eval_pred
        pred = pred[..., :2]
        labels = labels[..., :2]
        
        assert ( pred.sum(axis=1)<=1 ).sum() == pred.shape[0]
        
        pred = np.argmax(pred, axis=1)
        labels = np.argmax(labels, axis=1)
        
        res = {
            'F1': f1_score(labels, pred, average='binary', zero_division=0),
            # 'Acc': np.sum(pred*labels)/len(pred),
        }
        
        # for i, target_type in enumerate(self.label_list):
        #     res[target_type] = f1_score(pred[:,i], labels[:,i], zero_division=0)
        
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


class SubtextDiscriminateData(CustomData):
    def get_data_collator(self):
        return DataCollatorWithPadding(self.tokenizer)
    
    def get_compute_metrics(self):
        return SubtextDiscriminateComputeMetrics(self.label_list, self.tokenizer)

    def get_gt_vector(
        self, 
        df:pd.DataFrame,
    ):
        eye = np.eye(2)
        subtext_res = (df['subtext_res']!=0).astype(int)
        return eye[subtext_res]
    
    def get_dataset(
        self,
        split:Literal['train', 'dev', 'test',],
    ) -> CustomDataset:
        df = self.get_preprocessed_dataframe(split=split)
        df = df[df['subtext']!='']
        
        if split == 'train':
            return CustomDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(df=df, prompt=self.prompt['train_x'],
                                         tokenizer=self.tokenizer)),
                y_nums=self.get_gt_vector(df=df),
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
                y_nums=self.get_gt_vector(df=df),
                max_length=self.max_length,
            )
    