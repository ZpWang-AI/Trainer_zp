from .BaselineClassification import *
# from .BaselineGeneration import *
# from .CompareSimilarityModel import *
# from .CPKD import *
# from .MultiTaskModel import *
# from .PCPModel import *
# from .SubtextDiscriminatorModel import *
# from .SubtextGeneration import *
from .TestModel import *


MODEL_LIST = [
    BaselineClassificationModel,
    # BaselineGenerationModel,
    # CPKDModel,
    # MultitaskModel,
    # PCPModel,
    # SubtextDiscriminatorModel,
    # SubtextGenerationModel,
    TestModel,
]


def get_model_by_name(model_name:str):
    model_name = model_name.lower()
    for model in MODEL_LIST:
        if model_name == model.__name__.lower():
            return model
    raise Exception('wrong model_name')