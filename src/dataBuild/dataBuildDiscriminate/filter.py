from utils_zp.common_import import *
add_sys_path(__file__, 3)
add_sys_path(__file__, 2)

import numpy as np
import pandas as pd 
from sklearn.metrics import classification_report, confusion_matrix

from utils_zp import plt, plot_hist, visualize_1d_matrix, dump_json
from IDRR_data import IDRRDataFrames


class FilterMain:
    def __init__(self, target_csv, filter_func, fig_dir, column_name) -> None:
        fig_dir = path(fig_dir)
        make_path(dir_path=fig_dir)
        
        df = pd.read_csv(target_csv)
        # df = df[pd.notna(df['split'])]
        df = df[df['split']=='train']
        df = df[pd.notna(df['subtext_res'])]
        subtext_res = df['subtext_res'] == 1
        
        todo_data = df[column_name]
        plot_hist(todo_data, bins=20, fig_path=fig_dir/f'hist.png')
        visualize_1d_matrix(todo_data, fig_path=fig_dir/f'visual.png')

        filter_res = []
        for index, row in df.iterrows():
            filter_res.append(filter_func(row))
        
        confusion_mat = confusion_matrix(
            y_true=subtext_res, y_pred=filter_res,
            labels=[0,1], 
        )
        print(classification_report(
            y_true=subtext_res, y_pred=filter_res,
            labels=[0,1], zero_division=0, output_dict=False
        ))
        cls_report = classification_report(
            y_true=subtext_res, y_pred=filter_res,
            labels=[0,1], zero_division=0, output_dict=True
        )
        
        self.res = {
            'target csv': str(target_csv),
            'column name': column_name,
            'total_num': len(df),
            'filtered_num': sum(filter_res),
            'label1 precision': cls_report['1']['precision'],
            'confusion matrix': confusion_mat.tolist(),
            'cls report': cls_report,
        }


if __name__ == '__main__':
    cmp_sim_threshold = 0.6
    # cmp_sim_threshold = 0.9
    def cmp_sim_filter(row:pd.Series):
        return row['st_cmp_sim'] > cmp_sim_threshold

    discriminate_threshold = 0.66
    def discriminate_filter(row:pd.Series):
        return row['st_discriminate'] > discriminate_threshold
    
    save_dir = path(f'/data/zpwang/Trainer/log_space_main/filter_subtext/st_discriminate4_{discriminate_threshold}')
    res = FilterMain(
        target_csv='/data/zpwang/Trainer/data/dataBuild/subtext_discriminate4/pdtb3_l1_implicit.st_discriminate4.csv',
        filter_func=discriminate_filter,
        fig_dir=save_dir,
        column_name='st_discriminate',
    ).res
    res['threshold'] = discriminate_threshold
    res['desc'] = ''
    # print(res)
    print(dump_json(res, save_dir/'res.json', indent=4))