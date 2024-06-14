import os
import sys
import json
import shutil
import pandas as pd 
import numpy as np
import time
import random

from typing import *
from pathlib import Path as path
from tqdm import tqdm

sys.path.insert(0, str(path(__file__).parent.parent))

from utils_zp import dump_json, load_json
from IDRR_data import IDRRDataFrames, PromptFiller
from data import CustomDataset
from model import get_model_by_name, CustomModel


random.seed(2000)


class BuildCSV:
    def __init__(
        self,
        data_name,
        data_level,
        data_relation,
        data_path,
        hint_ratio,
        target_csv,
    ) -> None:
        dfs = IDRRDataFrames(
            data_name=data_name,
            data_level=data_level,
            data_relation=data_relation,
            data_path=data_path,
        )
        
        df_list = []
        
        for split in 'train dev test'.split():
            dfs.data_level = data_level
            cur_df = dfs.get_dataframe(split=split)
            hints = []
            for sense in cur_df['label11']:
                if random.random() < hint_ratio:
                    hints.append(f'Answer: {sense}')
                else:
                    hints.append('')
            dfs.data_level = 'raw'
            cur_df = dfs.get_dataframe(split=split)
            cur_df['hint'] = hints
            df_list.append(cur_df)

        res_df = pd.concat(df_list, axis=0, ignore_index=True,)
        res_df = res_df.drop('index', axis=1)
        res_df.to_csv(target_csv, index=False)
            
            
if __name__ == '__main__':
    BuildCSV(
        data_name='pdtb3',
        data_level='top',
        data_relation='Implicit',
        data_path='/data/zpwang/Trainer/data/used/pdtb3.p1.csv',
        hint_ratio=0.1,
        target_csv='/data/zpwang/Trainer/data/dataBuild/with_hint/pdtb3_l1_implicit.hint10.csv'
    )