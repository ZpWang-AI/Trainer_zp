from utils_zp.common_import import *
import shutil
import pandas as pd 
import numpy as np

sys.path.insert(0, str(path(__file__).parent.parent.parent))

from utils_zp import dump_json, load_json
from IDRR_data import IDRRDataFrames, PromptFiller
from data import CustomDataset
from model import get_model_by_name, CustomModel



class BuildCSV:
    def __init__(self, subtext_dirs, target_csv, column_name) -> None:
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
            # cur_df['subtext'] = results
            cur_df[column_name] = results
            df_list.append(cur_df)

        res_df = pd.concat(df_list, axis=0, ignore_index=True,)
        res_df = res_df.drop('index', axis=1)
        res_df.to_csv(target_csv, index=False)
            
            
if __name__ == '__main__':
    # BuildCSV([
    #     '/data/zpwang/Trainer/data/dataBuild/compare_similarity/pdtb3_dev_cmp_sim',
    #     '/data/zpwang/Trainer/data/dataBuild/compare_similarity/pdtb3_test_cmp_sim',
    #     '/data/zpwang/Trainer/data/dataBuild/compare_similarity/pdtb3_train_cmp_sim',
    # ], 
    #          target_csv='/data/zpwang/Trainer/data/dataBuild/compare_similarity/pdtb3_l1_implicit.cmp_sim.csv',
    #          column_name='st_cmp_sim')
    BuildCSV(
        [
            '/data/zpwang/Trainer/data/dataBuild/subtext_discriminate4/pdtb3_dev_subtext_distill_discriminate',
            '/data/zpwang/Trainer/data/dataBuild/subtext_discriminate4/pdtb3_test_subtext_distill_discriminate',
            '/data/zpwang/Trainer/data/dataBuild/subtext_discriminate4/pdtb3_train_subtext_distill_discriminate',
        ],
        target_csv='/data/zpwang/Trainer/data/dataBuild/subtext_discriminate4/pdtb3_l1_implicit.st_discriminate4.csv',
        column_name='st_discriminate'
    )