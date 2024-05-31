import os
import shutil
import time 
import datetime
import logging
import json
import traceback

from pathlib import Path as path
from typing import *


class CatchedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def catch_and_record_exception(exception_file, exception_info='', mode='a'):
    exception_string = '\n\n'.join(filter(None, map(lambda x:str(x).strip('\n'), [
        f"{'>'*10} ERROR {'>'*10}",
        str(datetime.datetime.today()),
        traceback.format_exc(),
        exception_info,
        f"{'<'*10} ERROR {'<'*10}",
    ])))+'\n\n'
    with open(exception_file, mode, encoding='utf8')as f:
        f.write(exception_string)
        print(exception_string)
        

if __name__ == '__main__':
    try:
        1/0
    except:
        catch_and_record_exception('exception.tmp')