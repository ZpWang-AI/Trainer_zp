import os
import sys
from pathlib import Path as path
sys.path.insert(0, str(path(__file__).parent.parent))

from utils.gpu_utils import GPUManager
CUDA_CNT = 1  
CUDA_ID = GPUManager.set_cuda_visible(target_mem_mb=24000, cuda_cnt=CUDA_CNT)

import json
import shutil
import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
import time
import transformers

from typing import *
from tqdm import tqdm
from transformers import (DataCollatorWithPadding, set_seed,
                          )

from utils import dump_json, load_json
from IDRR_data import DataFrames, DataFrames2, PromptFiller
from data import CustomDataset
from model import get_model_by_name, CustomModel


class GenerateMain:
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
        x_prompt,
        
        output_dir,
    ) -> None:
        hparams = load_json(hyperparams_path)
        model = get_model_by_name(hparams['model_name'])
        model_config = hparams['model_config']
        model_config['generation_config'] = {
            'max_length': 512,
            'max_new_tokens': 1000,
        }
        # transformers.GenerationConfig(
        #     max_length=512,
        #     max_new_tokens=1000,
        # )
        model = model(**model_config)
        model:CustomModel
        model.load_state_dict(torch.load(model_ckpt_path))
        model.to('cuda:0')
        model.eval()

        df = DataFrames2(
            data_name=data_name,
            label_level=data_level,
            relation=data_relation,
            data_path=data_path,
        ).get_dataframe(split=data_split)
        
        if x_prompt:
            pass
        elif 'x' in hparams['prompt']:
            x_prompt = hparams['prompt']['x']
        elif 'eval_x' in hparams['prompt']:
            x_prompt = hparams['prompt']['eval_x']
        else:
            raise 'lack of x_prompt'

        tokenizer = transformers.AutoTokenizer.from_pretrained(hparams['base_model_path'])
        dataset = CustomDataset(
            tokenizer=tokenizer,
            x_strs=list(PromptFiller(df=df, prompt=x_prompt, tokenizer=tokenizer)),
            y_strs=['a']*len(df),
        )
        datacollator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        with torch.no_grad():
            results = []
            for sp in tqdm(range(0, len(dataset), batch_size)):
                batch = datacollator([
                    dataset[p] for p in range(sp, min(sp+batch_size, len(dataset)))
                ])
                for k in batch:
                    batch[k] = batch[k].to('cuda:0')
                # print(batch)
                model_output = model.generate(**batch)
                decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)
                results.extend(decoded_output)
            
        output_dir = path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(hyperparams_path, output_dir/'hyperparams.json')
        generate_setting = {
            'model_ckpt_path': model_ckpt_path,
            'data_name': data_name,
            'data_level': data_level,
            'data_relation': data_relation,
            'data_path': data_path,
            'data_split': data_split,
            'batch_size': batch_size,
            'x_prompt': x_prompt,
        }
        dump_json(generate_setting, file_path=output_dir/'generate_setting.json', mode='w', indent=4)
        dump_json(results, file_path=output_dir/'generate_results.json', mode='w', indent=4)
        
    # def prepare_model
    
    
if __name__ == '__main__':
    for split in 'train dev test'.split():
        GenerateMain(
            hyperparams_path='/data/zpwang/IDRR_ConnT5/log_space_main/pdtb3_l1/subtext_distil/2024-05-22-17-56-51.pdtb3.level1.subtextdistil.base2.ep15_bs16_lr3e-05_flant5base/hyperparams.json',
            model_ckpt_path='/data/zpwang/IDRR_ConnT5/log_space_main/pdtb3_l1/subtext_distil/2024-05-22-17-56-51.pdtb3.level1.subtextdistil.base2.ep15_bs16_lr3e-05_flant5base/train_iter_0/checkpoint_best_Meteor/model.pth',
            data_name='pdtb3',
            data_level='level1',
            data_relation='Implicit',
            data_path='/data/zpwang/IDRR_ConnT5/data/used/pdtb3.p1.csv',
            data_split=split,
            batch_size=32,
            x_prompt='',
            output_dir=f'/data/zpwang/IDRR_ConnT5/data/subtext_distil2/pdtb3_{split}_subtext_distil2',
        )