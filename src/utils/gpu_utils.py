import os
import time
import pynvml
import threading

from typing import *
from utils import dump_json, load_json


def norm_mem_to_mb(mem):
    return int(mem) >> 20

    
class GPUManager:
    @staticmethod
    def query_gpu_memory(cuda_id:int, show=True):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info = {
            'cuda': cuda_id,
            'free': info.free,
            'used': info.used,
            'total': info.total,
        }
        pynvml.nvmlShutdown()
        if show:
            print(*[f'{k}: {norm_mem_to_mb(v)}MB' if k !='cuda' else f'cuda: {v}'
                    for k,v in info.items()], sep=', ')
        return info

    @staticmethod
    def get_all_cuda_id():
        pynvml.nvmlInit()
        cuda_cnt = list(range(pynvml.nvmlDeviceGetCount()))
        pynvml.nvmlShutdown()
        return cuda_cnt
    
    @staticmethod
    def get_free_gpus(
        gpu_cnt=1,
        target_mem_mb=8000,
        device_range=None,
        return_str=True,
        wait_seconds=5,
        show_waiting=False,
    ):
        if not device_range:
            device_range = GPUManager.get_all_cuda_id()
            # device_range = GPUManager.get_all_cuda_id()[::-1]

        target_mem_mb *= 1024**2
        while 1:
            gpu_id_lst = []         
            for cuda_id in device_range:
                if GPUManager.query_gpu_memory(cuda_id=cuda_id, show=False)['free'] > target_mem_mb:
                    gpu_id_lst.append(cuda_id)
                    if len(gpu_id_lst) >= gpu_cnt:
                        return ','.join(map(str,gpu_id_lst)) if return_str else gpu_id_lst
            if show_waiting:
                print('waiting cuda ...')
            time.sleep(wait_seconds)
    
    @staticmethod
    def set_cuda_visible(target_mem_mb=10000, cuda_cnt=1, device_range=None):
        free_cuda_ids = GPUManager.get_free_gpus(
            target_mem_mb=target_mem_mb,
            gpu_cnt=cuda_cnt,
            return_str=True,
            device_range=device_range,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = free_cuda_ids
        print(f'=== CUDA {free_cuda_ids} ===')
        return free_cuda_ids


# class GPUOccupier:
#     @staticmethod
#     def _occupy_one_gpu(cuda_id, target_mem_mb=8000):
#         import torch
#         '''
#         < release by following >
#         gpustat -cpu
#         kill -9 <num>
#         '''
#         device = torch.device(f'cuda:{cuda_id}')
#         used_mem = GPUManager.query_gpu_memory(cuda_id=cuda_id, show=False)[1]
#         used_mem_mb = used_mem/(1024**2)
#         one_gb = torch.zeros(224*1024**2)  # about 951mb
#         gb_cnt = int((target_mem_mb-used_mem_mb)/1024)
#         if gb_cnt < 0:
#             return
#         lst = [one_gb.detach().to(device) for _ in range(gb_cnt+1)]
#         while 1:
#             time.sleep(2**31)
            
#     @staticmethod
#     def wait_and_occupy_free_gpu(
#         target_mem_mb=8000,
#         wait_gap=5,
#         show_waiting=False,
#         device_range=None, 
#     ):
#         if not device_range:
#             device_range = GPUManager.get_all_cuda_id()
#         cuda_id = GPUManager.get_free_gpu(
#             target_mem_mb=target_mem_mb,
#             force=False,
#             wait=True,
#             wait_gap=wait_gap,
#             show_waiting=show_waiting,
#             device_range=device_range,
#         )
#         GPUManager._occupy_one_gpu(
#             cuda_id=cuda_id,
#             target_mem_mb=target_mem_mb,
#         )
        

class GPUMemoryMonitor:
    def __init__(self, cuda_ids:List[int], save_path, monitor_gap=3) -> None:
        self.cuda_ids = cuda_ids
        self.save_path = save_path
        self.monitor_gap = monitor_gap
        
        self.start_time = time.time()
        self.keep_monitor = True

        if cuda_ids:
            total_gpu_memory = GPUManager.query_gpu_memory(
                cuda_id=cuda_ids[0], show=False,
            )['total']
            total_gpu_memory = norm_mem_to_mb(total_gpu_memory)
            dump_json([0, total_gpu_memory], save_path, mode='a')
            self.process = threading.Thread(
                target=self.monitor, daemon=True,
            )
            self.process.start()
        
    def monitor(self):
        while self.keep_monitor:
            mems = [
                norm_mem_to_mb(
                    GPUManager.query_gpu_memory(
                        cuda_id=cid, show=False,
                    )['used']
                ) for cid in self.cuda_ids
            ]
            monitor_time = int(time.time()-self.start_time)/60

            dump_json([monitor_time, mems], self.save_path, mode='a')
            
            # self.gpu_memory.append(mem)
            # self.monitor_time.append(monitor_time)
            
            time.sleep(self.monitor_gap)
    
    def close(self):
        self.keep_monitor = False
        self.process.join()
        
    @classmethod
    def load_json_get_xy(cls, file_path) -> Tuple[List[float], List[List[int]]]:
        res = load_json(file_path)
        total_mem = res.pop(0)[1]
        if not res:
            raise Exception('wrong result')
        x = []
        ys = []
        for t, mems in res:
            x.append(t)
            ys.append(mems+[total_mem])
        ys = list(zip(*ys))
        return x, ys


if __name__ == '__main__':
    # print(GPUManager.query_gpu_memory(0))
    # print(GPUManager.get_free_gpus(target_mem_mb=2000))
    
    pass