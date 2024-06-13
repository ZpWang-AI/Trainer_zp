import os
import sys
import json
import shutil
import pandas as pd 
import numpy as np
import time
import random

from typing import *
from collections import *
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
        # data_name,
        # label_level,
        # data_relation,
        data_path,
        target_csv,
    ) -> None:
        df = pd.read_csv(data_path)
        for column_name in 'arg1 arg2 conn1 conn2'.split():
            df[column_name] = df[column_name].str.lower()
        df.to_csv(target_csv, index=False)
            
if __name__ == '__main__':
    BuildCSV(
        data_path='/data/zpwang/IDRR_ConnT5/data/used/pdtb2.p1.csv',
        target_csv='/data/zpwang/IDRR_ConnT5/data/used/pdtb2.p2.csv'
    )
    
    # df = pd.read_csv('/data/zpwang/IDRR_ConnT5/data/used/pdtb2.p1.csv')
    # print(Counter(df['conn2']))
    # for row in df.itertuples():
    #     conn = row.conn1
    #     if conn != conn.lower():
    #         print(conn)
    # for k in Counter(df['conn1']):
    #     if k != k.lower():
    #         print(k)