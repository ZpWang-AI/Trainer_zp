import os
import json
import numpy as np
import pandas as pd
import datetime

from typing import *
from collections import defaultdict
from pathlib import Path as path
from matplotlib import pyplot as plt

from utils_zp import (get_json_data_from_dir, GPUMemoryMonitor,
                   plot_curve, mark_extremum, dump_json)
# from main import Main


class Analyser:
    @classmethod
    def analyze_metrics(cls, log_dir, file_name, just_average=True):
        total_metrics = get_json_data_from_dir(root_dir=log_dir, file_name=file_name)
                    
        metric_analysis = {}
        for k, v in total_metrics.items():
            if just_average:
                metric_analysis[k] = np.mean(v)
            else:
                metric_analysis[k] = {
                    'tot': v,
                    'cnt': len(v),
                    'mean': np.mean(v),
                    'variance': np.var(v),
                    'std': np.std(v),
                    'error': np.std(v)/np.sqrt(len(v)),
                    'min': np.min(v),
                    'max': np.max(v),
                    'range': np.max(v)-np.min(v),
                }
        return metric_analysis
    
    @classmethod
    def draw_target_curve(
        cls, log_dir, file_name, 
        x_key, y_key, res_png, 
        mark_max=False, mark_min=False,
    ):
        log_dir = path(log_dir)
        for iter_dir in os.listdir(log_dir):
            iter_dir = log_dir/iter_dir
            if iter_dir.is_dir() and file_name in os.listdir(iter_dir):
                xs, ys = [], []
                with open(iter_dir/file_name, 'r', encoding='utf8')as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            line = json.loads(line)
                            if y_key not in line:
                                return
                            xs.append(line[x_key])                    
                            ys.append(line[y_key])
                if mark_max:
                    # ys = np.array(ys)*100
                    max_y = np.max(ys)
                    max_yid = np.argmax(ys)
                    plt.ylim(max_y-0.1, max_y+0.05)
                    plt.text(xs[max_yid], ys[max_yid], f'{ys[max_yid]*100:.2f}')
                plt.plot(xs, ys)
                plt.savefig(iter_dir/res_png)
                plt.close()    
                
    @classmethod
    def analyze_results(cls, log_dir:Union[path, str], log_filename_dict:dict):
        log_dir = path(log_dir)
        for json_file_name in log_filename_dict.values():
            if json_file_name == log_filename_dict['hyperparams']:
                continue
            metric_analysis = cls.analyze_metrics(log_dir, json_file_name, just_average=True)
            if metric_analysis:
                dump_json(metric_analysis, log_dir/json_file_name, indent=4)
                print(json.dumps(metric_analysis, indent=4))
        
        cls.draw_target_curve(
            log_dir=log_dir,
            file_name=log_filename_dict['loss'],
            x_key='epoch', y_key='loss', res_png='loss_curve.png'
        )
        for metric_name in 'Macro-F1 Meteor F1'.split():
            cls.draw_target_curve(
                log_dir=log_dir,
                file_name=log_filename_dict['dev'],
                res_png=f'dev_{metric_name}_curve.png',
                x_key='epoch', y_key='dev_'+metric_name, 
                mark_max=True,
            )
        
        mem_x, mem_ys = GPUMemoryMonitor.load_json_get_xy(
            file_path=log_dir/log_filename_dict['gpu_mem']
        )
        for mem_y in mem_ys:
            plt.plot(mem_x, mem_y)
            mark_extremum(mem_x, mem_y, mark_max=True)
        plt.savefig(log_dir/'gpu_mem_curve.png')
        plt.close()
        
    @classmethod
    def format_analysis_value(cls, value, format_metric=False, format_runtime=False, decimal_place=2):
        if not value:
            if format_metric:
                return '', ''
            return ''
        if format_metric:
            mean = '%.2f' % (value['mean']*100)
            error = '%.2f' % (value['error']*100)
            return mean, error
        elif format_runtime:
            return str(datetime.timedelta(seconds=int(value['mean'])))
        return f"{value['mean']:.{decimal_place}f}"

    @classmethod
    def analyze_experiment_results(
        cls,
        root_log_fold,
        target_csv_filename,
        hyperparam_keywords,
        hyperparam_filename,
        test_metric_filename,
        best_metric_filename,
        train_output_filename,
    ):
        from utils_zp import dict_to_defaultdict
        results = []
        for log_dir in os.listdir(root_log_fold):
            log_dir = path(root_log_fold, log_dir)
            cur_result = {}

            hyper_path = path(log_dir, hyperparam_filename)
            if hyper_path.exists():
                with open(hyper_path, 'r', encoding='utf8')as f:
                    hyperparams = json.load(f)
                for k, v in hyperparams.items():
                    if k in hyperparam_keywords:
                        cur_result[k] = v
            
            test_analysis = cls.analyze_metrics_json(log_dir, test_metric_filename)
            best_analysis = cls.analyze_metrics_json(log_dir, best_metric_filename)
            train_output_analysis = cls.analyze_metrics_json(log_dir, train_output_filename)
            test_analysis, best_analysis, train_output_analysis = map(dict_to_defaultdict, [
                test_analysis, best_analysis, train_output_analysis
            ])
            
            cur_result['acc'], cur_result['acc error'] = cls.format_analysis_value(test_analysis['test_Acc'], format_metric=True)
            cur_result['f1'], cur_result['f1 error'] = cls.format_analysis_value(test_analysis['test_Macro-F1'], format_metric=True)
            cur_result['epoch acc'] = cls.format_analysis_value(best_analysis['best_epoch_Acc'])
            cur_result['epoch f1'] = cls.format_analysis_value(best_analysis['best_epoch_Macro-F1'])
            cur_result['sample ps'] = cls.format_analysis_value(train_output_analysis['train_samples_per_second'])
            cur_result['runtime'] = cls.format_analysis_value(train_output_analysis['train_runtime'], format_runtime=True)
            
            results.append(cur_result)
        
        df_results = pd.DataFrame(results)
        # df_results.sort_values(by=['F1'], ascending=True)
        df_results.to_csv(target_csv_filename, encoding='utf-8')
        
        print(df_results)
    

if __name__ == '__main__':
    # root_log_fold = './experiment_results/epoch_and_lr'
    # target_csv_file = './experiment_results/epoch_and_lr.csv'
    # hyperparam_keywords = 'log_dir version learning_rate epochs'.split()
    # Analyser.analyze_experiment_results(
    #     root_log_fold = root_log_fold,
    #     target_csv_filename=target_csv_file,
    #     hyperparam_keywords=hyperparam_keywords,
    #     hyperparam_filename='hyperparams.json',
    #     test_metric_filename='test_metric_score.json',
    #     best_metric_filename='best_metric_score.json',
    #     train_output_filename='train_output.json',
    # )
    Analyser.analyze_results(
        log_dir='/data/zpwang/IDRR_ConnT5/log_space_main/2024-05-19-18-04-28.pdtb3.level1.subtextdistil.test.ep25_bs32_lr3e-05_ft5small',
        log_filename_dict = {
            'hyperparams': 'hyperparams.json',
            'best': 'best_metric_score.json',
            'dev': 'dev_metric_score.jsonl',
            'test': 'test_metric_score.json',
            'loss': 'train_loss.jsonl',
            'output': 'train_output.json',
            'gpu_mem': 'gpu_mem.jsonl',
        }
    )