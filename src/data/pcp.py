import json
import os
import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from typing import *
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score
    
from IDRR_data import DataFrames, PromptFiller, DataFrames2
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


class PCPData(CustomData):
    # train_input_y_nums = False
    # train_input_y_strs = True
    
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.dataframes = DataFrames2.from_DataFrames(self.dataframes)
        
    def get_data_collator(self):
        return DataCollatorWithPadding(self.tokenizer)
    
    def get_ans_word_vector(
        self, 
        df:pd.DataFrame,
        secondary_label_weight=0.,
    ):
        num_ans_word = len(self.dataframes.get_ans_word_list())
        eye = np.eye(num_ans_word+1, num_ans_word)
        primary_label_ids = df['conn1id'].astype(int)
        label_vector = eye[primary_label_ids]
        if secondary_label_weight:
            eye *= secondary_label_weight
            sec_label_ids = df['conn2id'].copy()
            sec_label_ids[pd.isna(sec_label_ids)] = num_ans_word
            sec_label_ids = sec_label_ids.astype(int)
            label_vector += eye[sec_label_ids]
        return label_vector
    
    def get_dataset(
        self,
        split:Literal['train', 'dev', 'test'],
    ) -> CustomDataset:
        df = self.get_preprocessed_dataframe(split=split)
        if split == 'train':
            return CustomDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(df=df, prompt=self.prompt['train_x'], 
                                         tokenizer=self.tokenizer)),
                y_nums=self.get_ans_word_vector(
                    df=df, secondary_label_weight=self.secondary_label_weight,
                ),
                max_length=self.max_length,
                shift_num=self.shift_num,
            )
        else:
            return CustomDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(
                    df=df, prompt=self.prompt['eval_x'], 
                    ignore=self.test_x_ignore, 
                    tokenizer=self.tokenizer,
                )),
                y_nums=self.get_label_vector(
                    df=df,
                    secondary_label_weight=1,
                ),
                max_length=self.max_length,
            )