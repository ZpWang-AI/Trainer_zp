import os
import json

from typing import *
from pathlib import Path as path

from utils.attr_dic import AttrDict


def fill_with_delimiter(s:str):
    return f'{"="*10} {s} {"="*(30-len(s))}'


class ExpArgs(AttrDict):
    def __init__(self) -> None:
        # ========== 'base_setting' ================
        self.part1 = 'base_setting'
        self.desc = 'test'
        
    @property
    def version(self):
        raise "TODO version"
    
    def format_part(self):    
        for p in range(1000):
            part_name = f'part{p}'
            if part_name in self:
                part_val:str = self[part_name]
                if not part_val.startswith('='*10):
                    self[part_name] = fill_with_delimiter(part_val)
    
    @staticmethod
    def format_part_in_file(file_path):
        with open(file_path, 'r', encoding='utf8')as f:
            lines = f.readlines()
        part_cnt = 1
        prefix_space = ' '*8
        target_lines:List[str] = []
        for line in lines:
            if line.strip().startswith('self.part'):
                part_val = line.split('=')[-1].strip()
                comment = f'{prefix_space}# {fill_with_delimiter(part_val)}\n'
                if target_lines[-1].strip()[0] == '#':
                    target_lines[-1] = comment
                else:
                    target_lines.append(comment)
                target_lines.append(f'{prefix_space}self.part{part_cnt} = {part_val}\n')
                part_cnt += 1
            else:
                target_lines.append(line)
        
        with open(file_path, 'w', encoding='utf8')as f:
            f.writelines(target_lines)
    

if __name__ == '__main__':
    # sample_args = CustomArgs(test_setting=False)
    # print(sample_args)
    
    # CustomArgs.format_args_part_in_file(__file__)
    pass