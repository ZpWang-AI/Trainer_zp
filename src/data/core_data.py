from utils_zp import *

import numpy as np
import pandas as pd
import transformers

from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from transformers.data.data_collator import DataCollator
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score


class CustomComputeMetrics:
    def __init__(self, label_list:list) -> None:
        self.metric_names = None
    
    def __call__(self, eval_pred):
        raise Exception()
    

class CustomDataCollator:
    # tokenizer
    def __init__(self, tokenizer):
        raise Exception()
    #     self.tokenizer = tokenizer
        
    def __call__(self, features, return_tensors=None):
        raise Exception()
    

class CustomDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        raise Exception()
    
    def __len__(self):
        raise Exception()


class CustomDataConfig(AttrDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.trainset_size = None
        self.devset_size = None
        self.testset_size = None
        
    def refill(self, data_:"CustomData"):
        for split in 'train dev test'.split():
            self[f'{split}set_size'] = len(data_.get_dataset(split))
            

class CustomData:
    def __init__(self, data_config:CustomDataConfig=None) -> None:
        self.data_config = data_config

        self.data_collator:CustomDataCollator = self.get_data_collator()
        self.compute_metrics:CustomComputeMetrics = self.get_compute_metrics()
    
    def get_data_collator(self):
        raise Exception()
    
    def get_compute_metrics(self):
        raise Exception()
        return CustomComputeMetrics(self.label_list)
    
    def get_dataset(
        self,
        split:Literal['train', 'dev', 'test'],
    ) -> CustomDataset:
        raise Exception()
            
    @property
    def train_dataset(self):
        return self.get_dataset(split='train')
    
    @property
    def dev_dataset(self):
        return self.get_dataset(split='dev')

    @property
    def test_dataset(self):
        return self.get_dataset(split='test')
    