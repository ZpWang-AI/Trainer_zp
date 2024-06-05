# ===== prepare server_name, root_fold =====
SERVER_NAME = 't2s'
if SERVER_NAME in ['cu13_', 'northern_']:
    ROOT_DIR = '/data/zpwang/IDRR_ConnT5/'
    PRETRAINED_MODEL_DIR = '/data/zpwang/pretrained_models/'
elif SERVER_NAME == 'cu12_':
    raise 
    ROOT_DIR = '/home/zpwang/IDRR/'
elif SERVER_NAME == 'SGA100':
    ROOT_DIR = '/public/home/hongy/zpwang/IDRR_ConnT5/'
    PRETRAINED_MODEL_DIR = '/public/home/hongy/pretrained_models/'
elif SERVER_NAME == 't2s':
    ROOT_DIR = '/home/qwe/test/zpwang/Trainer'
    PRETRAINED_MODEL_DIR = '/home/qwe/test/pretrained_model/'
else:
    raise Exception('wrong ROOT_DIR')

import os, sys
from pathlib import Path as path

BRANCH = 'main'
CODE_SPACE = ROOT_DIR+'src/'
DATA_SPACE = ROOT_DIR+'data/used/'
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CODE_SPACE)

# from arguments import CustomArgs
from utils.gpu_utils import GPUManager
# === TODO: prepare gpu ===
CUDA_CNT = 1  
CUDA_ID = GPUManager.set_cuda_visible(target_mem_mb=24000, cuda_cnt=CUDA_CNT)
# CUDA_ID = CustomArgs().prepare_gpu(target_mem_mb=10500, gpu_cnt=CUDA_CNT) 

# ===== import ===== 
from arguments import CustomArgs
from model import BaselineClassificationConfig
from main import Main


def base_experiment_args():
    args = CustomArgs()
    
    # ========== 'base setting' ================
    # args.part1 = 'base setting'
    args.task_name = 'classification'
    args.save_ckpt = False
    args.seed = 2023
    args.cuda_cnt = CUDA_CNT
    args.training_iteration = 5
    
    # ========== 'file path' ===================
    # args.part2 = 'file path'
    args.data_path = DATA_SPACE+'pdtb3.p1.csv'
    args.base_model_path = PRETRAINED_MODEL_DIR+'roberta-base'
    args.log_dir = ROOT_DIR+f'log_space_{BRANCH}'
    args.ckpt_dir = ''

    # ========== 'data' ========================
    # args.part3 = 'data'
    args.data_name = 'pdtb3'
    args.label_level = 'level1'
    args.data_relation = 'Implicit'
    args.prompt = {'x': 'Arg1: {arg1}\nArg2: {arg2}', 'y': '{conn1sense1}'}
    args.secondary_label_weight = 0.5
    args.mini_dataset = False
    args.data_augmentation_flatten_sec_label = False
    args.data_augmentation_add_conn_to_arg2 = False
    args.subtext_threshold = 0

    # args.trainset_size = -1
    # args.devset_size = -1
    # args.testset_size = -1
    
    # ========== 'model' =======================
    # args.part4 = 'model'
    args.model_name = 'baselineclassificationmodel'
    args.model_config = BaselineClassificationConfig(
        base_model_path=args.base_model_path,
        num_labels=4,
        loss_type='celoss'
    )
    
    # args.model_parameter_cnt = ''

    # ========== 'optimizer' ===================
    # args.part5 = 'optimizer'
    args.weight_decay = 0.01
    args.learning_rate = 3e-5
    
    # ========== 'epoch, batch, step' ==========
    # args.part6 = 'epoch, batch, step'
    args.max_steps = -1
    args.warmup_ratio = 0.05
    args.epochs = 25
    args.train_batch_size = 32
    args.eval_batch_size = 32
    args.eval_steps = 100
    args.log_steps = 10
    args.gradient_accumulation_steps = 1
    args.eval_per_epoch = 5

    # args.real_batch_size = -1
    # args.eval_samples = -1
    
    # ========== 'additional details' ==========
    # args.part7 = 'additional details'
    args.cuda_id = CUDA_ID
    # args.create_time = ''
    args.server_name = SERVER_NAME
    
    # **************************************************************
    # **************************************************************
    
    args.epochs = 25
    args.train_batch_size = 32
    args.learning_rate = 3e-5
    args.prompt = {'x': '{arg1} <sep> {arg2}'}
    args.secondary_label_weight = 0.5
    args.base_model = 'roberta-base'
    
    args.desc = 'basetest'
    args._version_info_list = [
        args.create_time,
        args.data_name,
        args.label_level,
        args.task_name,
        args.desc,
        f'ep{args.epochs}_bs{args.train_batch_size}_lr{args.learning_rate}_secl{args.secondary_label_weight}',
        args.base_model
    ]
    return args
    
    
if __name__ == '__main__':
    Main(base_experiment_args())
    
    pass
