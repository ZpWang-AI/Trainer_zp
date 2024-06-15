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

from utils_zp import (catch_and_record_exception, AttrDict, dump_json,
                      count_parameters, GPUMemoryMonitor, set_process_title)
from arguments import CustomArgs
from IDRR_data import IDRRDataFrames
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
        # set_process_title('Trainer')
        self.main_one_round(args)
    
    def main_one_round(self, args:CustomArgs):
        from copy import deepcopy

        args.ckpt_dir = path(args.ckpt_dir)/args.version
        args.log_dir = path(args.log_dir)/args.version
        args.check_path()
        args.format_part()
        
        # === data ===
        data_class = get_data_by_name(args.task_name)
        # data = CustomData(
        data = data_class(
            data_path=args.data_path,
            data_name=args.data_name,
            data_level=args.data_level,
            data_relation=args.data_relation,
            
            base_model_path=args.base_model_path,
            prompt=args.prompt,

            max_length=args.max_input_length,
            secondary_label_weight=args.secondary_label_weight,
            mini_dataset=args.mini_dataset,
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
        if (
            isinstance(data.tokenizer, transformers.RobertaTokenizer) or
            isinstance(data.tokenizer, transformers.RobertaTokenizerFast)
        ):
            args.fill_model_config(
                ans_word_list_token=data.dataframes.get_ans_word_token_id_list(tokenizer=data.tokenizer),
                ans_label_list_id=data.dataframes.ans_lid_list,
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
        
        # === hyperparams ===
        args.dump_json(args.log_dir/self.log_filename_dict['hyperparams'])
        print(args)
        
        try:
            for _training_iter_id in range(args.training_iteration):
                self.main_one_iteration(
                    deepcopy(args),
                    data=data,
                    model=model,
                    training_iter_id=_training_iter_id,
                )
        except Exception:
            error_file = args.log_dir/'error.out'
            catch_and_record_exception(error_file)
            exit(1)
        
        gpu_monitor.close()
        Analyser.analyze_results(args.log_dir, self.log_filename_dict)
        
        print('='*20+'\nDone')
        
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
        args.ckpt_dir /= train_fold_name
        args.log_dir /= train_fold_name
        args.check_path()
        
        if training_iter_id:
            model.initial_model()
        
        args.dump_json(args.log_dir/self.log_filename_dict['hyperparams'])
        
        # === fit ===
        
        start_time = time.time()
        
        self.fit(
            args=args,
            # training_args=training_args,
            data=data,
            model=model,
            # trainer_class=get_trainer_by_name(args.task_name),
        )

        total_runtime = time.time()-start_time
        with open(args.log_dir/self.log_filename_dict['output'], 'r', encoding='utf8')as f:
            train_output = json.load(f)
            train_output['total_runtime'] = total_runtime
        dump_json(train_output, args.log_dir/self.log_filename_dict['output'], indent=4)

        if not args.save_ckpt:
            if args.ckpt_dir == args.log_dir:
                for log_path in os.listdir(args.log_dir):
                    if 'checkpoint' in log_path or log_path == 'runs':
                        shutil.rmtree(path(args.log_dir)/log_path)
            else:
                shutil.rmtree(args.ckpt_dir)
           
    def fit(
        self,
        args:CustomArgs, 
        data:CustomData, 
        model:CustomModel, 
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
            metric_names=compute_metrics.metric_names,
        )
        callback.best_metric_file_name = args.log_dir/self.log_filename_dict['best']
        callback.dev_metric_file_name = args.log_dir/self.log_filename_dict['dev']
        callback.train_loss_file_name = args.log_dir/self.log_filename_dict['loss']
        
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
        dump_json(train_output, args.log_dir/self.log_filename_dict['output'])
        
        # === do test ===
        callback.evaluate_testdata = True
        test_metrics = self.test(
            trainer=trainer,
            model=model,
            metric_names=compute_metrics.metric_names,
            ckpt_dir=args.ckpt_dir,
            test_dataset=data.test_dataset,
            log_filename=self.log_filename_dict['test']
        )
        dump_json(test_metrics, args.log_dir/self.log_filename_dict['test'], indent=4)
        print(json.dumps(test_metrics, indent=4))
        return trainer, callback
    
    def test(
        self,
        trainer:Trainer,
        model:CustomModel, 
        test_dataset:CustomDataset,
        metric_names:List[str],
        ckpt_dir:str,
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
        return test_metrics
 
