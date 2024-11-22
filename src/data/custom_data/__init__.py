from .IDRR import *
from .test_data import *


data_list = [
    ClassificationData,
    CPKDData,
    GenerationData,
    MultitaskData,
    PCPData,
    SubtextDiscriminateData,
    SubtextDistilData,
    TestData,
]


def get_data_by_name(task_name:str):
    task_name = task_name.lower()
    for data in data_list:
        if data.__name__.lower().startswith(task_name):
            return data
    raise Exception('wrong task_name')