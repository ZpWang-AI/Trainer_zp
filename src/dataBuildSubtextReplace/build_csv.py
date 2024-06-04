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

from utils import dump_json, load_json
from IDRR_data import DataFrames, DataFrames2, PromptFiller
from data import CustomDataset
from model import get_model_by_name, CustomModel


random.seed(2000)


class BuildCSV:
    def __init__(
        self,
        data_name,
        # label_level,
        data_relation,
        data_path,
        # hint_ratio,
        subtext_threshold,
        target_csv,
    ) -> None:
        dfs = DataFrames2(
            data_name=data_name,
            label_level='raw',
            relation=data_relation,
            data_path=data_path,
        )
        
        df_list = []
        
        for split in 'train dev test'.split():
            x_input = []
            cur_df = dfs.get_dataframe(split=split)
            if subtext_threshold == 1:
                for index, row in cur_df.iterrows():
                    if row.subtext_res == 1:
                        x_input.append(f'{row.subtext}')
                    else:
                        x_input.append(f'{row.arg1} <sep> {row.arg2}')
            elif subtext_threshold == 2:
                for index, row in cur_df.iterrows():
                    if row.subtext_res != 0:
                        x_input.append(f'{row.subtext}')
                    else:
                        x_input.append(f'{row.arg1} <sep> {row.arg2}')
            else:
                raise 'wrong subtext_threshold'

            dfs.label_level = 'raw'
            cur_df = dfs.get_dataframe(split=split)
            cur_df['x_input'] = x_input
            df_list.append(cur_df)

        res_df = pd.concat(df_list, axis=0, ignore_index=True,)
        res_df = res_df.drop('index', axis=1)
        path(target_csv).parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(target_csv, index=False)
            
            
if __name__ == '__main__':
    BuildCSV(
        data_name='pdtb3',
        # label_level='level1',
        data_relation='Implicit',
        data_path='/home/qwe/test/zpwang/Trainer/data/used/pdtb3_l1_implicit.subtext.csv',
        subtext_threshold=2,
        target_csv='/home/qwe/test/zpwang/Trainer/data/dataBuild/subtext_replace/pdtb3_l1_implicit.subtext_replace.csv'
    )