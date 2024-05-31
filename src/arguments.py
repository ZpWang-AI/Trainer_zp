import os
import json

from typing import *
from pathlib import Path as path
from datetime import datetime

from utils import AttrDict
from IDRR_data import DataFrames


def fill_with_delimiter(s:str):
    return f'{"="*10} {s} {"="*(30-len(s))}' if not s.startswith('='*10) else s


class CustomArgs(AttrDict):
    def __init__(self) -> None:
        self.version = 'init'
        
        # ========== 'base setting' ================
        self.part1 = 'base setting'
        self.task_name = 'classification'
        self.save_ckpt = False
        self.seed = 2023
        self.cuda_cnt = 1
        self.training_iteration = 5
        self.bf16 = False
        self.fp16 = False
        
        # ========== 'file path' ===================
        self.part2 = 'file path'
        self.data_path = '/public/home/hongy/zpwang/IDRR_ConnT5/data/used/pdtb3.p1.csv'
        self.base_model_path = 'roberta-base'
        self.log_dir = '/content/drive/MyDrive/IDRR/log_space'
        self.ckpt_dir = ''

        # ========== 'data' ========================
        self.part3 = 'data'
        self.data_name = 'pdtb3'
        self.label_level = 'level1'
        self.data_relation = 'Implicit'
        self.prompt = {'x': 'Arg1: {arg1}\nArg2: {arg2}', 'y': '{conn1sense1}'}
        self.max_input_length = 512
        self.secondary_label_weight = 0.5
        self.mini_dataset = False
        self.data_augmentation_flatten_sec_label = False
        self.data_augmentation_add_conn_to_arg2 = False
        self.subtext_threshold = 0

        self.trainset_size = -1
        self.devset_size = -1
        self.testset_size = -1
        
        # ========== 'model' =======================
        self.part4 = 'model'
        self.model_name = 'baselineclassificationmodel'
        self.model_config:dict = None
        
        self.base_model = ''
        self.model_parameter_cnt = ''

        # ========== 'optimizer' ===================
        self.part5 = 'optimizer'
        self.weight_decay = 0.01
        self.learning_rate = 3e-5
        
        # ========== 'epoch, batch, step' ==========
        self.part6 = 'epoch, batch, step'
        self.max_steps = -1
        self.warmup_ratio = 0.05
        self.epochs = 25
        self.train_batch_size = 32
        self.eval_batch_size = 32
        self.eval_steps = 100
        self.log_steps = 10
        self.gradient_accumulation_steps = 1
        self.eval_per_epoch = 5

        self.real_batch_size = -1
        self.eval_samples = -1
        
        # ========== 'additional details' ==========
        self.part7 = 'additional details'
        self.cuda_id = ''
        self.server_name = ''
        self.create_time = ''

        self.justify_part()
        
    def justify_part(self):    
        for p in range(1000):
            attr_name = f'part{p}'
            if hasattr(self, attr_name):
                init_attr = self.__getattribute__(attr_name)
                self.__setattr__(attr_name, fill_with_delimiter(init_attr))
    
    def fill_model_config(self, **kwargs):
        for key in self.model_config:
            if key in kwargs:
                self.model_config[key] = kwargs[key]
        
    # def estimate_cuda_memory(self):
    #     return 30000
    
    # def prepare_gpu(self, target_mem_mb=10000, gpu_cnt=None):
    #     if not self.cuda_id:
    #         if target_mem_mb < 0:
    #             target_mem_mb = self.estimate_cuda_memory()
    #         if gpu_cnt is None:
    #             gpu_cnt = self.cuda_cnt

    #         from utils import GPUManager
    #         free_gpu_ids = GPUManager.get_free_gpus(
    #             gpu_cnt=gpu_cnt, 
    #             target_mem_mb=target_mem_mb,
    #         )
    #         os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu_ids
    #         self.cuda_id = free_gpu_ids
    #         print(f'=== CUDA {free_gpu_ids} ===')
    #     return self.cuda_id

    def complete_path(self, show_create_time=True, specific_info:Union[list, tuple]=None):
        if not self.create_time:
            self.set_create_time()
            
            specific_fold_name = []
            if show_create_time:
                specific_fold_name.append(self.create_time)
            if specific_info:
                specific_fold_name.extend(specific_info)
            specific_fold_name.append(self.version)
            specific_fold_name = '.'.join(map(str, specific_fold_name))
            
            self.log_dir = os.path.join(self.log_dir, specific_fold_name) 
            if not self.ckpt_dir:
                self.ckpt_dir = self.log_dir
            else:
                self.ckpt_dir = os.path.join(self.ckpt_dir, specific_fold_name)
            
    def check_path(self):
        if not self.create_time:
            print('===\nwarning: auto complete path\n===')
            self.complete_path()
        
        assert path(self.data_path).exists(), 'wrong data path'
        assert path(self.base_model_path).exists(), 'wrong model path'
        path(self.log_dir).mkdir(parents=True, exist_ok=True)
        path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
    
    def recalculate_eval_log_steps(self):
        self.real_batch_size = self.train_batch_size \
                             * self.gradient_accumulation_steps \
                             * self.cuda_cnt
        if self.eval_per_epoch > 0:
            self.eval_steps = int(self.trainset_size / self.eval_per_epoch / self.real_batch_size)
            self.log_steps = self.eval_steps // 10
            self.eval_steps = max(1, self.eval_steps)
            self.log_steps = max(1, self.log_steps)
        self.eval_samples = self.real_batch_size * self.eval_steps
        

if __name__ == '__main__':
    # sample_args = CustomArgs(test_setting=False)
    # print(sample_args)
    
    def format_args_part():
        with open(__file__, 'r', encoding='utf8')as f:
            lines = f.readlines()
        part_cnt = 1
        prefix_space = ' '*8
        for p, line in enumerate(lines):
            if line.strip().startswith('self.part'):
                part_name = line.split('=')[-1].strip()
                lines[p-1] = f'{prefix_space}# {fill_with_delimiter(part_name)}\n'
                lines[p] = f'{prefix_space}self.part{part_cnt} = {part_name}\n'
                part_cnt += 1
        
        with open(__file__, 'w', encoding='utf8')as f:
            f.writelines(lines)
    
    format_args_part()