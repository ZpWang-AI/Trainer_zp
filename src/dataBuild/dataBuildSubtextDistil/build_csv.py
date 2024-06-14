import os
import sys
import json
import shutil
import pandas as pd 
import numpy as np
import time

from typing import *
from pathlib import Path as path
from tqdm import tqdm

sys.path.insert(0, str(path(__file__).parent.parent))

from utils_zp import dump_json, load_json
from IDRR_data import IDRRDataFrames, PromptFiller
from data import CustomDataset
from model import get_model_by_name, CustomModel



class BuildCSV:
    def __init__(self, subtext_dirs, target_csv) -> None:
        df_list = []
        
        for subtext_dir in subtext_dirs:
            subtext_dir = path(subtext_dir)
            results = load_json(subtext_dir/'generate_results.json')
            setting = load_json(subtext_dir/'generate_setting.json')
            hparams = load_json(subtext_dir/'hyperparams.json')
            
            cur_df = IDRRDataFrames(
                data_name=setting['data_name'],
                data_level='raw',
                data_relation=setting['data_relation'],
                data_path=setting['data_path'],
            ).get_dataframe(split=setting['data_split'])
            cur_df['subtext'] = results
            cur_df['subtext_res'] = [1]*len(results)
            df_list.append(cur_df)

        res_df = pd.concat(df_list, axis=0, ignore_index=True,)
        res_df.to_csv(target_csv, index=False)
            
            
if __name__ == '__main__':
    BuildCSV([
        '/data/zpwang/Trainer/data/subtext_distil/pdtb3_train_subtext_distil',
        '/data/zpwang/Trainer/data/subtext_distil/pdtb3_dev_subtext_distil',
        '/data/zpwang/Trainer/data/subtext_distil/pdtb3_test_subtext_distil',
    ], target_csv='/data/zpwang/Trainer/data/subtext_distil/res.csv')