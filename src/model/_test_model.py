from utils_zp import *
add_sys_path(__file__, 2)

import torch
from transformers import AutoTokenizer

from model import *
from utils_zp.ml import count_parameters


def test_model(sample_model, sample_tokenizer):
    print(sample_model)
    print('param num:', count_parameters(sample_model))
    print('='*30)
    
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
    logits = torch.tensor([[0.8, 0.5, 0.9, 0.4, 0.7],
                        [0.3, 0.6, 0.1, 0.7, 0.5]])
    y_pred = torch.softmax(logits, dim=1)
    y_true = torch.tensor([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0]])       
    y_true2 = torch.tensor([0, 1]) 
    criterion1 = CELoss()
    criterion2 = nn.CrossEntropyLoss(reduction='mean')
    loss1 = criterion1(y_pred, y_true)
    loss2 = criterion2(logits, y_true2)
    print(loss1, loss2, sep='\n')


if __name__ == '__main__':
    # demo_CELoss()
    base_model_path = '/public/home/hongy/pretrained_models/flan-t5-large/'
    base_model_path = r'D:\pretrained_models\roberta-base'
    # model = AutoModel.from_pretrained('roberta-base', cache_dir=r'D:\pretrained_models', output_loading_info=True)
    # model, loading_info = transformers.RobertaForMaskedLM.from_pretrained('roberta-base', output_loading_info=True)
    model, loading_info = transformers.BertForPreTraining.from_pretrained('bert-base-uncased', output_loading_info=True)
    test_model(
        sample_model=BaselineClassificationModel(base_model_path=base_model_path, num_labels=4, loss_type='CELoss'), 
        sample_tokenizer=AutoTokenizer.from_pretrained(base_model_path),
    )
    