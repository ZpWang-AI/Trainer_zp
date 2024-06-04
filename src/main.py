import os 
import json
import shutil
import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
import time
import transformers

from typing import *
from pathlib import Path as path
from transformers import (TrainingArguments, Trainer, DataCollatorWithPadding, set_seed,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig,
                          AutoModelForSeq2SeqLM)

from utils import (catch_and_record_exception,
                   CustomLogger, count_parameters, GPUMemoryMonitor)
from arguments import CustomArgs
from IDRR_data import DataFrames2
from data import CustomData, CustomDataset, get_data_by_name, CustomComputeMetrics
from model import get_model_by_name, BaselineClassificationConfig, BaselineGenerationConfig, MultitaskConfig, CustomModel
# from trainer import CustomTrainer, get_trainer_by_name
from callbacks import CustomCallback
from analyze import Analyser

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


'''
group:
    make experiments with different hyperparam settings
round:
    get the mean results of iterations
iteration:
    train
    eval on dev dataset every some steps
    eval on test dataset by the best ckpt 
epoch, batch/step:
    ...
'''

class Main:
    log_filename_dict = {
        'hyperparams': 'hyperparams.json',
        'best': 'best_metric_score.json',
        'dev': 'dev_metric_score.jsonl',
        'test': 'test_metric_score.json',
        'loss': 'train_loss.jsonl',
        'output': 'train_output.json',
        'gpu_mem': 'gpu_mem.jsonl',
    }
    
    def __init__(self, args:CustomArgs) -> None:
        args.complete_path(
            show_create_time=True,
            specific_info=(
                args.data_name, 
                args.label_level,
                args.task_name,
            )
        )
        self.main_one_round(args)        
        pass
    
    def fit(
        self,
        args:CustomArgs, 
        # training_args:TrainingArguments, 
        logger:CustomLogger,
        data:CustomData, 
        model:CustomModel, 
        # trainer_class,
    ):
        training_args = Seq2SeqTrainingArguments(
            output_dir = args.ckpt_dir,
            
            # strategies of evaluation, logging, save
            evaluation_strategy = "steps", 
            eval_steps = args.eval_steps,
            logging_strategy = 'steps',
            logging_steps = args.log_steps,
            save_strategy = 'no',
            
            # optimizer and lr_scheduler
            optim = 'adamw_torch',
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay,
            lr_scheduler_type = 'linear',
            warmup_ratio = args.warmup_ratio,
            
            # epochs and batches 
            num_train_epochs = args.epochs, 
            max_steps = args.max_steps,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            
            # train consumption
            eval_accumulation_steps=10,
            bf16=args.bf16,
            fp16=args.fp16,
            
            # generation_config=GenerationConfig(
            #     num_beams=1, do_sample=True, num_retrun_sequences=3,
            # ),
            # data_seed=args.seed,
            predict_with_generate=True,
            # deepspeed={},
            # accelerator_config={}
            report_to='none',
            auto_find_batch_size=False,
        )
        
        compute_metrics = data.compute_metrics
        callback = CustomCallback(
            logger=logger, 
            metric_names=compute_metrics.metric_names,
        )
        callback.best_metric_file_name = self.log_filename_dict['best']
        callback.dev_metric_file_name = self.log_filename_dict['dev']
        callback.train_loss_file_name = self.log_filename_dict['loss']
        
        trainer = Seq2SeqTrainer(
            model=model, 
            args=training_args, 
            tokenizer=data.tokenizer, 
            compute_metrics=compute_metrics,
            callbacks=[callback],
            data_collator=data.data_collator,
            
            train_dataset=data.train_dataset,
            eval_dataset=data.dev_dataset, 
        )
        callback.trainer = trainer

        train_output = trainer.train().metrics
        # train_output = {}
        logger.log_json(train_output, self.log_filename_dict['output'], log_info=True)
        
        # === do test ===
        callback.evaluate_testdata = True
        test_metrics = self.test(
            trainer=trainer,
            model=model,
            metric_names=compute_metrics.metric_names,
            ckpt_dir=args.ckpt_dir,
            test_dataset=data.test_dataset,
            logger=logger,
            log_filename=self.log_filename_dict['test']
        )
                    
        return trainer, callback
    
    def test(
        self,
        trainer:Trainer,
        model:CustomModel, 
        test_dataset:CustomDataset,
        metric_names:List[str],
        ckpt_dir:str,
        logger:CustomLogger,
        log_filename:str,
    ):
        test_metrics = {}
        for metric_ in metric_names:
            load_ckpt_dir = path(ckpt_dir)/f'checkpoint_best_{metric_}'
            if load_ckpt_dir.exists():
                if (load_ckpt_dir/'pytorch_model.bin').exists():
                    model.load_state_dict(torch.load(load_ckpt_dir/'pytorch_model.bin'))
                # elif (load_ckpt_dir/'model.safetensors').exists():
                    # model.load_state_dict(torch.load(load_ckpt_dir/'model.safetensors')['model_state_dict'])
                    # pass
                elif (load_ckpt_dir/'model.pth').exists():
                    model.load_state_dict(torch.load(load_ckpt_dir/'model.pth'))
                else:
                    raise Exception('wrong ckpt path')
                evaluate_output = trainer.evaluate(eval_dataset=test_dataset)
                test_metrics['test_'+metric_] = evaluate_output['eval_'+metric_]
        logger.log_json(test_metrics, log_file_name=log_filename, log_info=True)
        return test_metrics
    
    def main_one_iteration(
        self,
        args:CustomArgs, 
        data:CustomData,
        model:CustomModel,
        training_iter_id=0
    ):
        # === prepare === 
        # seed
        args.seed += training_iter_id
        set_seed(args.seed)
        data.shift_num = args.seed
        # path
        train_fold_name = f'train_iter_{training_iter_id}'
        args.ckpt_dir = os.path.join(args.ckpt_dir, train_fold_name)
        args.log_dir = os.path.join(args.log_dir, train_fold_name)
        args.check_path()
        
        logger = CustomLogger(
            log_dir=args.log_dir,
            logger_name=f'{args.create_time}_iter{training_iter_id}_logger',
            stream_handler=True,
        )
        
        if training_iter_id:
            model.initial_model()
        
        logger.log_json(dict(args), self.log_filename_dict['hyperparams'], log_info=False)

        # === fit ===
        
        start_time = time.time()
        
        self.fit(
            args=args,
            # training_args=training_args,
            logger=logger,
            data=data,
            model=model,
            # trainer_class=get_trainer_by_name(args.task_name),
        )

        total_runtime = time.time()-start_time
        with open(logger.log_dir/self.log_filename_dict['output'], 'r', encoding='utf8')as f:
            train_output = json.load(f)
            train_output['total_runtime'] = total_runtime
        logger.log_json(train_output, self.log_filename_dict['output'], False)
        
        if not args.save_ckpt:
            if args.ckpt_dir == args.log_dir:
                for log_path in os.listdir(args.log_dir):
                    if 'checkpoint' in log_path or log_path == 'runs':
                        shutil.rmtree(path(args.log_dir)/log_path)
            else:
                shutil.rmtree(args.ckpt_dir)
            
    def main_one_round(self, args:CustomArgs):
        from copy import deepcopy

        args.check_path()
        args.justify_part()
        
        # === data ===
        data_class = get_data_by_name(args.task_name)
        # data = CustomData(
        data = data_class(
            data_path=args.data_path,
            data_name=args.data_name,
            label_level=args.label_level,
            relation=args.data_relation,
            
            base_model_path=args.base_model_path,
            prompt=args.prompt,

            max_length=args.max_input_length,
            secondary_label_weight=args.secondary_label_weight,
            mini_dataset=args.mini_dataset,
            data_augmentation_flatten_sec_label=args.data_augmentation_flatten_sec_label,
            data_augmentation_add_conn_to_arg2=args.data_augmentation_add_conn_to_arg2,
            subtext_threshold=args.subtext_threshold,
        )
        data: CustomData
        args.trainset_size, args.devset_size, args.testset_size = map(len, [
            data.train_dataset, data.dev_dataset, data.test_dataset
        ])
        args.recalculate_eval_log_steps()
        args.fill_model_config(
            base_model_path=args.base_model_path,
            num_labels=data.num_labels,
            label_list=data.label_list,
            loss_type='CELoss',
        )
        if isinstance(data.dataframes, DataFrames2) and (
            isinstance(data.tokenizer, transformers.RobertaTokenizer) or
            isinstance(data.tokenizer, transformers.RobertaTokenizerFast)
        ):
            args.fill_model_config(
                ans_word_list_token=data.dataframes.get_ans_word_list(tokenizer=data.tokenizer),
                ans_label_list_id=data.dataframes.get_ans_label_list(use_label_id=True),
            )
        
        # === gpu mem ===
        cuda_id_nums = [int(p.strip())for p in args.cuda_id.split(',')]
        gpu_monitor = GPUMemoryMonitor(
            cuda_ids=cuda_id_nums, 
            save_path=path(args.log_dir)/self.log_filename_dict['gpu_mem'],
            monitor_gap=3,
        )
        
        # === model ===
        model = get_model_by_name(args.model_name)
        model = model(**args.model_config)
        args.model_parameter_cnt = count_parameters(model)
        
        # === logger ===
        main_logger = CustomLogger(args.log_dir, logger_name=f'{args.create_time}_main_logger', stream_handler=True) 
        main_logger.log_json(dict(args), log_file_name=self.log_filename_dict['hyperparams'], log_info=True)
        
        try:
            for _training_iter_id in range(args.training_iteration):
                self.main_one_iteration(
                    deepcopy(args),
                    data=data,
                    model=model,
                    training_iter_id=_training_iter_id,
                )
        except Exception:
            error_file = main_logger.log_dir/'error.out'
            catch_and_record_exception(error_file)
            exit(1)
        
        gpu_monitor.close()
        Analyser.analyze_results(main_logger, self.log_filename_dict)


def local_test_args_classify(data_name='pdtb2', label_level='level1'):
    args = CustomArgs()
    args.cuda_cnt = 0
    
    # args.mini_dataset = True
    args.task_name = 'classification'
    args.secondary_label_weight = 0
    args.data_augmentation_add_conn_to_arg2 = False
    args.training_iteration = 2
    args.train_batch_size = 8
    args.eval_batch_size = 8
    args.epochs = 2
    args.eval_steps = 1
    args.log_steps = 1
    args.eval_per_epoch = -1
    args.prompt = {'x': 'Arg1: {arg1}\nArg2: {conn1} {arg2}', 'y': '{conn1sense1}'}
    
    args.version = 'test'
    args.server_name = 'local'
    
    # args.data_name = data_name
    # if data_name == 'pdtb2':
    #     args.data_path = '/public/home/hongy/zpwang/IDRR_ConnT5/data/used/pdtb2.p1.csv'
    # elif data_name == 'pdtb3':
    #     args.data_path = '/public/home/hongy/zpwang/IDRR_ConnT5/data/used/pdtb3.context2_2.p1.csv'
    # elif data_name == 'conll':
    #     args.data_path = '/public/home/hongy/zpwang/IDRR_ConnT5/data/used/conll.p1.csv'  
    args.data_name = 'pdtb3'
    args.data_path = '/public/home/hongy/zpwang/IDRR_ConnT5/data/used/pdtb3_test.p1.csv'
    args.label_level = label_level  
    
    args.base_model_path = '/public/home/hongy/pretrained_models/flan-t5-small'
    args.base_model_path = '/public/home/hongy/pretrained_models/roberta-base'
    args.log_dir = '/public/home/hongy/zpwang/IDRR_ConnT5/log_space/'
    args.ckpt_dir = ''
    
    args.model_name = 'baselineclassificationmodel'
    args.model_config = BaselineClassificationConfig(
        # loss_type=
    )
    
    # args.train_batch_size = 2
    # args.eval_steps = 5

    return args

def local_test_args_generate():
    args = local_test_args_classify()
    
    args.task_name = 'generation'
    args.base_model_path = '/public/home/hongy/pretrained_models/flan-t5-small'
    args.model_name = 'baselinegenerationmodel'
    args.model_config = BaselineGenerationConfig(
        # base_model_path=
        generation_config={
            'max_new_tokens':10,
            'do_sample': True,
            'num_return_sequences': 3,
        },
    )
    return args

def local_test_args_multitask():
    args = local_test_args_classify()
    args.task_name = 'multitask'
    args.base_model_path = '/public/home/hongy/pretrained_models/flan-t5-small'
    args.model_name = 'multitaskmodel'
    args.model_config = MultitaskConfig(
        # base_model_path=
        loss_type='celoss',
    )
    return args


if __name__ == '__main__':
    
    m = Main()
    m.main_one_round(local_test_args_classify())
    m.main_one_round(local_test_args_generate())
    m.main_one_round(local_test_args_multitask())