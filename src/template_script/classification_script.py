from script_head import *
from utils_zp import get_info_from_script_file


def experiment_args():
    args:CustomArgs = base_experiment_args()
    
    args.task_name = 'classification'
    args.training_iteration = 3
    args.save_ckpt = False

    args.data_name = 'pdtb3'
    args.data_path = '/home/qwe/test/zpwang/Trainer/data/used/pdtb3.p1.csv'
    args.prompt = {
        'x': '{arg1} <sep> {arg2}',
    }
    args.subtext_threshold = 0
    args.secondary_label_weight = 0.5
    args.epochs = 25
    args.train_batch_size = 32
    args.learning_rate = 2e-5
    
    args.gradient_accumulation_steps = 1
    args.eval_per_epoch = 5
    args.eval_batch_size = 16
    
    args.base_model = 'roberta-base'
    args.base_model_path = PRETRAINED_MODEL_DIR+args.base_model
    args.model_name = 'baselineclassificationmodel'
    from model import BaselineClassificationConfig
    args.model_config = BaselineClassificationConfig()
    
    script_info = get_info_from_script_file(__file__)
    args.desc, script_id = script_info['desc'], script_info['id']
    args._version_info_list = [
        args.create_time,
        args.data_name,
        args.data_level,
        script_id,
        args.desc,
        f'ep{args.epochs}_bs{args.train_batch_size}_lr{args.learning_rate}',
        args.base_model
    ]
    
    return args
    
    
if __name__ == '__main__':
    Main(experiment_args())
    