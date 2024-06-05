import os
import json
import logging
import datetime

from pathlib import Path as path
from collections import defaultdict
    

class StreamLogger:
    def __init__(self, logger_name='') -> None:
        if not logger_name:
            logger_name = str(datetime.datetime.now())
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
    
    def info(self, *args, sep=' ', end='\n'):
        self.logger.info(sep.join(map(str, args))+end)


# stream_logger = StreamLogger()
# def log_print(*args, sep=' ', end='\n', file=None, flush=False):
#     stream_logger.info(*args, sep=sep, end=end)


class CustomLogger:
    def __init__(self, log_dir='./log_space', logger_name='custom_logger', stream_handler=False) -> None:
        self.log_dir = path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        fh = logging.FileHandler(self.log_dir/'log.out')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        if stream_handler:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
    
    def info(self, *args):
        self.logger.info(' '.join(map(str, args)))
        
    def log_json(self, content, log_file_name, log_info=False, mode='w'):
        if log_info:
            self.logger.info('\n'+json.dumps(content, ensure_ascii=False, indent=4))
        
        log_file = path(self.log_dir)/log_file_name
        with open(log_file, mode=mode, encoding='utf8')as f:
            if mode == 'w':
                json.dump(content, f, ensure_ascii=False, indent=4)
            elif mode == 'a':
                json.dump(content, f, ensure_ascii=False)
                f.write('\n')
            else:
                raise ValueError('wrong mode')


if __name__ == '__main__':
    sample_logger = CustomLogger(log_dir='./tmp/', stream_handler=True)
    sample_logger.info('123', {'1231':1231})