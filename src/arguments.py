from utils_zp.common_import import *

from utils_zp import AttrDict, ExpArgs, make_path


class CustomArgs(ExpArgs):
    def __init__(self) -> None:
        self.desc = 'init'
        
        # ========== 'base setting' ================
        self.part1 = 'base setting'
        self.task_name = 'classification'
        self.save_ckpt = False
        self.seed = 2023
        self.cuda_cnt = 1
        self.training_iteration = 5
        
        # ========== 'file path' ===================
        self.part2 = 'file path'
        self.data_path = '/public/home/hongy/zpwang/Trainer/data/used/pdtb3.p1.csv'
        self.base_model_path = 'roberta-base'
        self.log_dir:path = '/content/drive/MyDrive/IDRR/log_space'
        self.ckpt_dir:path = ''

        # ========== 'data' ========================
        self.part3 = 'data'
        self.data_name = 'pdtb3'
        self.data_level = 'top'
        self.data_relation = 'Implicit'
        self.prompt = {'x': 'Arg1: {arg1}\nArg2: {arg2}', 'y': '{label11}'}
        self.max_input_length = 512
        self.secondary_label_weight = 0.5
        self.mini_dataset = False
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

        # ========== 'trainer' ===================
        self.part5 = 'trainer'
        self.weight_decay = 0.01
        self.learning_rate = 3e-5
        self.bf16 = False
        self.fp16 = False
        
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

        self.format_part()
        self.set_create_time()
        
        self._version_info_list =[
            self.create_time,
            self.data_name,
            self.data_level,
            self.task_name,
            self.desc,
        ]
        
    @property
    def version(self):
        return '.'.join(self._version_info_list)
        
    # def estimate_cuda_memory(self):
    #     return 30000
    
    # def prepare_gpu(self, target_mem_mb=10000, gpu_cnt=None):
    #     if not self.cuda_id:
    #         if target_mem_mb < 0:
    #             target_mem_mb = self.estimate_cuda_memory()
    #         if gpu_cnt is None:
    #             gpu_cnt = self.cuda_cnt

    #         from utils_zp import GPUManager
    #         free_gpu_ids = GPUManager.get_free_gpus(
    #             gpu_cnt=gpu_cnt, 
    #             target_mem_mb=target_mem_mb,
    #         )
    #         os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu_ids
    #         self.cuda_id = free_gpu_ids
    #         print(f'=== CUDA {free_gpu_ids} ===')
    #     return self.cuda_id
    
    def fill_model_config(self, **kwargs):
        if not isinstance(self.model_config, AttrDict):
            self.model_config = AttrDict.from_dict(self.model_config)
        self.model_config.merge_dict(kwargs, overwrite_existing=False)

    def check_path(self):
        assert path(self.data_path).exists(), 'wrong data path'
        assert path(self.base_model_path).exists(), 'wrong model path'
        make_path(dir_path=self.log_dir)
        make_path(dir_path=self.ckpt_dir)
    
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
    CustomArgs.format_part_in_file(__file__)