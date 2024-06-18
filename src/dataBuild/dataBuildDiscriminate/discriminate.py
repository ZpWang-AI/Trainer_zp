from utils_zp.common_import import *
add_sys_path(__file__, 3)

from utils_zp.gpu_utils import GPUManager
CUDA_CNT = 1  
CUDA_ID = GPUManager.set_cuda_visible(target_mem_mb=24000, cuda_cnt=CUDA_CNT)

import shutil
import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
import transformers

from transformers import DataCollatorWithPadding

from utils_zp import dump_json, load_json
from IDRR_data import IDRRDataFrames, PromptFiller
from data import CustomDataset
from model import get_model_by_name, CustomModel


class DiscriminateMain:
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
        device = 'cuda:0'
        
        hparams = load_json(hyperparams_path)
        model = get_model_by_name(hparams['model_name'])
        model_config = hparams['model_config']
        model = model(**model_config)
        model:CustomModel
        model.load_state_dict(torch.load(model_ckpt_path))
        model.to(device)
        model.eval()

        df = IDRRDataFrames(
            data_name=data_name,
            data_level=data_level,
            data_relation=data_relation,
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
            # y_nums=torch.zeros((len(df),4))
            # y_strs=['a']*(len(df))
        )
        datacollator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        with torch.no_grad():
            results = []
            for sp in tqdm.trange(0, len(dataset), batch_size):
                batch = datacollator([
                    dataset[p] for p in range(sp, min(sp+batch_size, len(dataset)))
                ])
                for k in batch:
                    batch[k] = batch[k].to(device)
                # print(batch)
                model_output = model.get_logits(**batch)
                model_output = torch.softmax(model_output, dim=1)
                # output = output.tolist()
                results.extend(model_output[...,1].cpu().numpy())
                
        results = list(map(float, results))
            
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
    for split in 'dev test train'.split():
    # for split in 'dev'.split():
        DiscriminateMain(
            hyperparams_path='/data/zpwang/Trainer/log_space_main.old/pdtb3_l1/subtext_discriminate/2024-05-28-19-57-21.pdtb3.level1.subtextdiscriminate.lrbs.ep25_bs8_lr2e-05_robertabase/train_iter_2/hyperparams.json',
            model_ckpt_path='/data/zpwang/Trainer/log_space_main.old/pdtb3_l1/subtext_discriminate/2024-05-28-19-57-21.pdtb3.level1.subtextdiscriminate.lrbs.ep25_bs8_lr2e-05_robertabase/train_iter_2/checkpoint_best_F1/model.pth',
            data_name='pdtb3',
            data_level='top',
            data_relation='Implicit',
            data_path='/data/zpwang/Trainer/data/used/pdtb3_l1_implicit.subtext.csv',
            data_split=split,
            batch_size=32,
            x_prompt='',
            output_dir=f'/data/zpwang/Trainer/data/dataBuild/subtext_discriminate4/pdtb3_{split}_subtext_distill_discriminate',
        )