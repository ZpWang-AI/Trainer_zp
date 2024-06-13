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
from IDRR_data import DataFrames, DataFrames2, PromptFiller
from data import CustomDataset
from model import get_model_by_name, CustomModel


random.seed(2000)


class BuildCSV:
    def __init__(
        self,
        data_name,
        # label_level,
        # data_relation,
        data_path,
        explicit_ratio_train,
        explicit_ratio_eval,
        target_csv,
    ) -> None:
        
        dfs = DataFrames2(
            data_name=data_name,
            label_level='raw',
            relation='All',
            data_path=data_path,
        )
        
        df_list = []
        
        for split in 'train dev test'.split():
            dfs.relation = 'Explicit'
            cur_df = dfs.get_dataframe(split=split)
            if split == 'train':
                cur_n = len(cur_df)
                explicit_data_id = random.sample(
                    range(cur_n), int(cur_n*explicit_ratio_train)
                )
                explicit_data = cur_df.iloc[explicit_data_id]
            else:
                cur_n = len(cur_df)
                explicit_data_id = random.sample(
                    range(cur_n), int(cur_n*explicit_ratio_eval)
                )
                explicit_data = cur_df.iloc[explicit_data_id]
            
            dfs.relation = 'Implicit'
            cur_df = dfs.get_dataframe(split=split)
            df_list.append(cur_df)
            df_list.append(explicit_data)

        path(target_csv).parent.mkdir(parents=True, exist_ok=True)
        res_df = pd.concat(df_list, axis=0, ignore_index=True,)
        res_df = res_df.drop('index', axis=1)
        res_df.to_csv(target_csv, index=False)
            
            
if __name__ == '__main__':
    BuildCSV(
        data_name='pdtb3',
        data_path='/data/zpwang/IDRR_ConnT5/data/used/pdtb3.p1.csv',
        explicit_ratio_train=0.2,
        explicit_ratio_eval=0,
        target_csv='/data/zpwang/IDRR_ConnT5/data/dataBuild/mix_explicit/pdtb3_l1.mix_explicit20.csv'
    )