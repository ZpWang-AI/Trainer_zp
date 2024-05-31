import sys, os
from pathlib import Path as path
sys.path.insert(0, str(path(__file__).parent.parent))

import torch
import time
from transformers import AutoTokenizer

from model import *
from utils import count_parameters


def test_model(sample_model):
    print(sample_model)
    print('param num:', count_parameters(sample_model))
    print('='*30)
    
    sample_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    sample_x = ['你好']*2+['hello world. Nice to see you']*3
    sample_x_token = sample_tokenizer(sample_x, padding=True, return_tensors='pt',)
    sample_y = torch.Tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0.5, 0, 0],
        [1, 0.5, -1, 0],
        [1, -0.5, 0, 0]
    ])
    
    start_time = time.time()
    sample_output = sample_model(sample_x_token['input_ids'], sample_x_token['attention_mask'], sample_y)
    for k, v in sample_output.items():
        print(k, v.shape) if k != 'loss' else print(k, v)
    print(f'train time: {time.time()-start_time}')
    print('='*30)

    sample_model.eval()
    with torch.no_grad():
        start_time = time.time()
        sample_output = sample_model(sample_x_token['input_ids'], sample_x_token['attention_mask'], sample_y)
        for k, v in sample_output.items():
            print(k, v.shape) if k != 'loss' else print(k, v)
        print(f'eval time: {time.time()-start_time}')
    print('='*30)


def demo_CELoss():
    y_pred = torch.tensor([[0.8, 0.5, 0.9, 0.4, 0.7],
                        [0.3, 0.6, 0.1, 0.7, 0.5]])
    y_true = torch.tensor([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0]])       
    y_true2 = torch.tensor([0, 1]) 
    criterion1 = CELoss()
    criterion2 = nn.CrossEntropyLoss(reduction='mean')
    loss1 = criterion1(y_pred, y_true)
    loss2 = criterion2(y_pred, y_true2)
    print(loss1, loss2, sep='\n')


if __name__ == '__main__':
    base_model_path = '/public/home/hongy/pretrained_models/flan-t5-large/'
    test_model(BaselineT5Model(base_model_path=base_model_path, num_labels=4, loss_type='CELoss'))