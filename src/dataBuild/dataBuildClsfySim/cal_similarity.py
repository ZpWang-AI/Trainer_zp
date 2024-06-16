from utils_zp.common_import import *
sys.path.insert(0, str(path(__file__).parent.parent.parent))

from utils_zp.gpu_utils import GPUManager
CUDA_CNT = 1  
CUDA_ID = GPUManager.set_cuda_visible(target_mem_mb=14000, cuda_cnt=CUDA_CNT)

import shutil
import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
import transformers

from tqdm import tqdm
from transformers import (DataCollatorWithPadding, set_seed,
                          )

from utils_zp import dump_json, load_json, make_path
from IDRR_data import IDRRDataFrames, PromptFiller
from data import CustomDataset
from model import (get_model_by_name, CustomModel, CompareSimilarityModel,
                   CompareSimilarityConfig)


class CalSimilarityMain:
    def __init__(
        self,
        hyperparams_path,
        model_ckpt_path,
        
        data_name,
        data_level,
        data_relation,
        data_path,
        data_split,
        
        batch_size,
        prompt_arg,
        prompt_subtext,
        
        output_dir,
    ) -> None:
        device = f'cuda:{CUDA_ID}'
        device = f'cuda:0'
        hparams = load_json(hyperparams_path)
        model_config = CompareSimilarityConfig.from_dict(hparams['model_config'])
        model_config.ft_model_path = model_ckpt_path
        model = CompareSimilarityModel(**model_config)
        model.to(device)
        model.eval()

        df = IDRRDataFrames(
            data_name=data_name,
            data_level=data_level,
            data_relation=data_relation,
            data_path=data_path,
        ).get_dataframe(split=data_split)

        tokenizer = transformers.AutoTokenizer.from_pretrained(hparams['base_model_path'])
        dataset_arg = CustomDataset(
            tokenizer=tokenizer,
            x_strs=PromptFiller(df=df, prompt=prompt_arg, tokenizer=tokenizer).list,
        )
        dataset_subtext = CustomDataset(
            tokenizer=tokenizer,
            x_strs=PromptFiller(df=df, prompt=prompt_subtext, tokenizer=tokenizer).list
        )
        sample_num = len(dataset_arg)
        datacollator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        with torch.no_grad():
            results = []
            for sp in tqdm(range(0, sample_num, batch_size)):
                batch_arg = datacollator([
                    dataset_arg[p] for p in range(sp, min(sp+batch_size, sample_num))
                ])
                batch_subtext = datacollator([
                    dataset_subtext[p] for p in range(sp, min(sp+batch_size, sample_num))
                ])
                for k in batch_arg:
                    batch_arg[k] = batch_arg[k].to(device)      
                for k in batch_subtext:
                    batch_subtext[k] = batch_subtext[k].to(device)
                # print(batch)
                model_output:torch.Tensor = model(batch_arg, batch_subtext)
                results.extend(model_output.cpu().numpy())
        
        results = list(map(float, results))
        
        output_dir = path(output_dir)
        make_path(dir_path=output_dir)
        shutil.copy(hyperparams_path, output_dir/'hyperparams.json')
        generate_setting = {
            'model_ckpt_path': model_ckpt_path,
            'data_name': data_name,
            'data_level': data_level,
            'data_relation': data_relation,
            'data_path': data_path,
            'data_split': data_split,
            'batch_size': batch_size,
            'prompt_arg': prompt_arg,
            'prompt_subtext': prompt_subtext,
        }
        dump_json(generate_setting, file_path=output_dir/'generate_setting.json', mode='w', indent=4)
        dump_json(results, file_path=output_dir/'generate_results.json', mode='w', indent=4)
        
    # def prepare_model
    
    
if __name__ == '__main__':
    for split in 'dev test train'.split():
    # for split in ['test']:
        CalSimilarityMain(
            hyperparams_path='/data/zpwang/Trainer/log_space_main/2024-06-15-19-23-08.pdtb3.top.001.plain.ep25_bs32_lr2e-05.roberta-base/train_iter_1/hyperparams.json',
            model_ckpt_path='/data/zpwang/Trainer/log_space_main/2024-06-15-19-23-08.pdtb3.top.001.plain.ep25_bs32_lr2e-05.roberta-base/train_iter_1/checkpoint_best_Macro-F1/model.pth',
            data_name='pdtb3',
            data_level='top',
            data_relation='Implicit',
            data_path='/data/zpwang/Trainer/data/used/pdtb3_l1_implicit.subtext.csv',
            data_split=split,
            batch_size=32,
            prompt_arg='{arg1}<sep>{arg2}',
            prompt_subtext='{subtext}',
            output_dir=f'/data/zpwang/Trainer/data/dataBuild/compare_similarity/pdtb3_{split}_cmp_sim',
        )