import json
import os
import numpy as np
import pandas as pd
import transformers

from collections import defaultdict, Counter
from typing import *
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score
    
from IDRR_data import DataFrames, PromptFiller, DataFrames2
from data import CustomDataCollator, CustomDataset, CustomData, CustomComputeMetrics
from data.pcp import PCPData

from utils_zp import format_element_to_shape


# class GenerationComputeMetrics(CustomComputeMetrics):
#     def __init__(self, label_list:list, tokenizer, label_to_id_func) -> None:
#         self.label_list = label_list
#         self.tokenizer = tokenizer
#         self.label_to_id_func = label_to_id_func
#         self.vote_num = 1
        
#         self.metric_names = ['Macro-F1', 'Acc']+label_list
#         self.metric_names = ['Macro-F1']
    

    
class CPKDDataCollator(CustomDataCollator):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator_padding = DataCollatorWithPadding(self.tokenizer)
        # self.data_collator_seq2seq = DataCollatorForSeq2Seq(self.tokenizer)
        pass
        
    def __call__(self, features):
        student_inputs = []
        teacher_inputs = []
        for feature in features:
            teacher_inputs.append(feature['teacher_inputs'])
            del feature['teacher_inputs']
            student_inputs.append(feature)
        student_inputs = self.data_collator_padding(student_inputs)
        teacher_inputs = self.data_collator_padding(teacher_inputs)
        batch = student_inputs
        batch['teacher_input_ids'] = teacher_inputs['input_ids']
        batch['teacher_attention_mask'] = teacher_inputs['attention_mask']
        return batch


class CPKDDataset(CustomDataset):
    def __init__(
        self, 
        tokenizer:transformers.PreTrainedTokenizer,
        x_strs:List[str], 
        teacher_x_strs:List[str]=None,
        y_nums:List[List[float]]=None,
        max_length:int=512,
        shift_num:int=0,
        # **kwargs
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            x_strs=x_strs,
            y_nums=y_nums,
            max_length=max_length,
            shift_num=shift_num,
        )
        
        self.teacher_x_strs = teacher_x_strs
        
    def __getitem__(self, index):
        assert 0 <= index < self.n
        index = (index+self.shift_num) % self.n
        model_inputs = self.token_encode(self.x_strs[index])
        teacher_inputs = self.token_encode(self.teacher_x_strs[index])
        model_inputs['teacher_inputs'] = teacher_inputs
        model_inputs['labels'] = self.y_nums[index]
    
        return model_inputs



class CPKDData(PCPData):
    # train_input_y_nums = False
    # train_input_y_strs = True
    
    def get_data_collator(self):
        return CPKDDataCollator(self.tokenizer)
        
    def get_dataset(
        self,
        split:Literal['train', 'dev', 'test',],
    ) -> CustomDataset:
        df = self.get_preprocessed_dataframe(split=split)
        if split == 'train':
            return CPKDDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(
                    df=df, prompt=self.prompt['train_x'], 
                    tokenizer=self.tokenizer
                )),
                teacher_x_strs=list(PromptFiller(
                    df=df, prompt=self.prompt['teacher_x'],
                    tokenizer=self.tokenizer,
                )),
                y_nums=self.get_ans_word_vector(
                    df=df, secondary_label_weight=self.secondary_label_weight,
                ),
                max_length=self.max_length,
                shift_num=self.shift_num,
            )
        else:
            return CPKDDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(
                    df=df, prompt=self.prompt['eval_x'], 
                    ignore=self.test_x_ignore, 
                    tokenizer=self.tokenizer,
                )),
                teacher_x_strs=['']*df.shape[0],
                y_nums=self.get_label_vector(
                    df=df,
                    secondary_label_weight=1,
                ),
                max_length=self.max_length,
            )