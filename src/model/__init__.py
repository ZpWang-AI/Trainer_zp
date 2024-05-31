from model.base_class import *
from model.BaselineClassification import *
from model.BaselineGeneration import *
from model.CPKD import *
from model.MultiTaskModel import *
from model.PCPModel import *
from model.SubtextDiscriminatorModel import *
from model.SubtextGeneration import *
from model.TestModel import *

from model.criterion import *
# from model.configs import *


model_list = [
    BaselineClassificationModel,
    BaselineGenerationModel,
    CPKDModel,
    MultitaskModel,
    PCPModel,
    SubtextDiscriminatorModel,
    SubtextGenerationModel,
    TestModel,
]


def get_model_by_name(model_name:str):
    model_name = model_name.lower()
    for model in model_list:
        if model_name == model.__name__.lower():
            return model
    raise Exception('wrong model_name')

