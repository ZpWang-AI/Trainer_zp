from data.base_class import *
from data.classification import *
from data.cpkd import *
from data.generation import *
from data.multitask_data import *
from data.pcp import *
from data.subtextDiscriminate import *
from data.subtextDistil import *
from data.test_data import *

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
