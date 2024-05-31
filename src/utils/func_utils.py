import datetime, time
import json

from collections import defaultdict


def clock_decorator(func):
    def new_func(*args, **kwargs):
        print(f'{func.__name__} starts')
        start_time = time.time()
        res = func(*args, **kwargs)
        running_time = time.time()-start_time
        running_time = running_time if running_time < 3 else int(running_time)
        running_time = datetime.timedelta(seconds=running_time)
        print(f'{func.__name__} ends, runtime: {running_time}')
        return res
    return new_func


def round_dict_values(dic, k):
    return {
        key: f'{value:.{k}f}' if isinstance(value, float) else value
        for key, value in dic.items()
    }
    

def dict_to_defaultdict(dic, default_type=dict):
    new_dic = defaultdict(default_type)
    new_dic.update(dic)
    return new_dic


def format_element_to_shape(target, json_indent=True):
    # import torch
    # import numpy as np
    if hasattr(target, 'shape'):
        res = str(target.shape)
    elif isinstance(target, dict):
        res = {k:format_element_to_shape(v, False)for k,v in target.items()}
    else:
        try:
            target = list(target)
            res = [format_element_to_shape(p, False)for p in target]
        except:
            raise Exception(f'wrong type {type(target)}')
    return json.dumps(res, indent=4) if json_indent else res
    
    
if __name__ == '__main__':
    @clock_decorator
    def f():
        time.sleep(3)
        
    
    f()