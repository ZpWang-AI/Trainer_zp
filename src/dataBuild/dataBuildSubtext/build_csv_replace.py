from utils_zp.common_import import *
import shutil
import random

sys.path.insert(0, str(path(__file__).parent.parent))

from utils_zp import dump_json, load_json, make_path
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
        prompt_default,
        prompt_subtext,
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
                        cur_prompt = prompt_subtext
                    else:
                        cur_prompt = prompt_default
                    x_input.append(PromptFiller.fill_prompt(row, cur_prompt))
            elif subtext_threshold == 2:
                    if row.subtext_res != 0:
                        cur_prompt = prompt_subtext
                    else:
                        cur_prompt = prompt_default
                    x_input.append(PromptFiller.fill_prompt(row, cur_prompt))
            else:
                raise 'wrong subtext_threshold'

            dfs.label_level = 'raw'
            cur_df = dfs.get_dataframe(split=split)
            cur_df['x_input'] = x_input
            df_list.append(cur_df)

        res_df = pd.concat(df_list, axis=0, ignore_index=True,)
        res_df = res_df.drop('index', axis=1)
        make_path(file_path=target_csv)
        res_df.to_csv(target_csv, index=False)
            
            
if __name__ == '__main__':
    BuildCSV(
        data_name='pdtb3',
        # label_level='level1',
        data_relation='Implicit',
        data_path='/home/qwe/test/zpwang/Trainer/data/used/pdtb3_l1_implicit.subtext.csv',
        prompt_default='Argument 1:\n{arg1}\n\nArgument 2:\n{arg2}\n\nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nA. Comparison\nB. Contingency\nC. Expansion\nD. Temporal\n\nAnswer:',
        prompt_subtext='Implicit meaning:\n{subtext}\n\nQuestion: Based on the implicit meaning, what is the discourse relation between arguments?\nA. Comparison\nB. Contingency\nC. Expansion\nD. Temporal\n\nAnswer:',
        subtext_threshold=1,
        target_csv='/home/qwe/test/zpwang/Trainer/data/dataBuild/subtext_replace/pdtb3_l1_implicit.subtext_replace.csv'
    )