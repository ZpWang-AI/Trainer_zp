from utils_zp import *


class CustomArgs(ExpArgs):
    def __init__(self) -> None:
        self.desc = 'init'
        
        # =========== base setting ===============
        self.part1 = 'base setting'
        self.save_ckpt = False
        self.seed = 2023
        self.cuda_cnt = 1
        self.training_iteration = 5
        
        # =========== filepath ==================
        self.part2 = 'filepath'
        self.log_dir:path = '/content/drive/MyDrive/IDRR/log_space'
        self.ckpt_dir:path = ''
        # self.data_path = '/public/home/hongy/zpwang/Trainer/data/used/pdtb3.p1.csv'
        # self.base_model_path = 'roberta-base'

        # =========== data =======================
        self.part3 = 'data'
        self.data_name = 'classification'
        self.data_config:dict = {}
        # self.data_level = 'top'
        # self.data_relation = 'Implicit'
        # self.prompt = {'x': 'Arg1: {arg1}\nArg2: {arg2}', 'y': '{label11}'}
        # self.max_input_length = 512
        # self.secondary_label_weight = 0.5
        # self.mini_dataset = False
        # self.subtext_threshold = 0

        # self.trainset_size = -1
        # self.devset_size = -1
        # self.testset_size = -1
        
        # =========== model ======================
        self.part4 = 'model'
        self.model_name = 'baselineclassificationmodel'
        self.model_config:dict = {}
        
        # self.base_model = ''
        # self.model_parameter_cnt = ''

        # =========== trainer ====================
        self.part5 = 'trainer'
        self.weight_decay = 0.01
        self.learning_rate = 3e-5
        self.bf16 = False
        self.fp16 = False
        
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
        
        # =========== additional details =========
        self.part7 = 'additional details'
        self.cuda_id = ''
        self.server_name = ''
        self.create_time:Datetime_ = ''

        self.format_part()
        self.set_create_time()
        
        self._version_info_list =[
            self.create_time.format_str(2),
            self.data_name,
            self.model_name,
            self.desc,
        ]

    def check_path(self):
        make_path(dir_path=self.log_dir)
        make_path(dir_path=self.ckpt_dir)
    
    def recalculate_eval_log_steps(self):
        self.real_batch_size = self.train_batch_size \
                             * self.gradient_accumulation_steps \
                             * self.cuda_cnt
        if self.eval_per_epoch > 0 and 'trainset_size' in self.data_config:
            self.eval_steps = int(self.data_config['trainset_size'] / self.eval_per_epoch / self.real_batch_size)
            self.log_steps = self.eval_steps // 10
            self.eval_steps = max(1, self.eval_steps)
            self.log_steps = max(1, self.log_steps)
        self.eval_samples = self.real_batch_size * self.eval_steps
        

if __name__ == '__main__':
    CustomArgs().format_part_in_file(__file__)