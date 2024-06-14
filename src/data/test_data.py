import json
import os
import numpy as np
import pandas as pd

from typing import *
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score
    
from IDRR_data import PromptFiller
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


class TestDataset(CustomDataset):
    pass
    def __init__(
        self, 
        tokenizer,
        x_strs:List[str], 
        y_nums:List[Union[float, List[float]]]=None,
        y_strs=None,
        # **kwargs
    ) -> None:
        super().__init__(tokenizer=tokenizer, x_strs=x_strs)
        
        self.tokenizer = tokenizer
        self.x_strs = x_strs
        # self.y_nums = y_nums
        
    def __getitem__(self, index):
        model_inputs = self.tokenizer(
            self.x_strs[index],
            add_special_tokens=True, 
            padding=True,
            truncation='longest_first', 
            max_length=1024,
        )
        # model_inputs['labels'] = self.tokenizer(
        #     self.y_strs[index],
        #     add_special_tokens=True, 
        #     padding=True,
        #     truncation='longest_first', 
        #     max_length=256,
        # ).input_ids
        model_inputs['labels'] = [index]
        
        # for key in self.extra_kwargs:
        #     model_inputs[key] = self.extra_kwargs[key][index]
    
        return model_inputs
    
    def __len__(self):
        return len(self.x_strs)


class TestData(CustomData):
    train_input_y_nums = True
    train_input_y_strs = False
    
    def get_data_collator(self):
        return DataCollatorForSeq2Seq(self.tokenizer)
    
    def get_dataset(
        self,
        split:Literal['train', 'dev', 'test', 'blind_test'],
    ) -> CustomDataset:
        df = self.get_preprocessed_dataframe(split=split)
        if split == 'train':
            return TestDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(df=df, prompt=self.prompt['x'])),
            )
        else:
            return TestDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(
                    df=df, prompt=self.prompt['x'], 
                    ignore=self.test_x_ignore
                )),
            )