from utils_zp import *
add_sys_path(__file__, 2)

from data import *


def test_data(dataclass, prompt):
    sample = dataclass(
        # data_path='/public/home/hongy/zpwang/Trainer/data/used/pdtb3.context2_2.p1.csv',
        # data_name='pdtb3',
        data_name='pdtb3',
        data_path='/public/home/hongy/zpwang/Trainer/data/used/pdtb3_test.p1.csv',
        data_level='second',
        data_relation='Implicit',
        base_model_path='/public/home/hongy/pretrained_models/roberta-base',
        prompt=prompt,
        mini_dataset=True,
    )
    print()
    print(sample.train_dataset[0])
    print(sample.test_dataset[0])
    print('='*20)
    pass


if __name__ == '__main__':
    test_data(ClassificationData, prompt='{arg1}\n{conn1} {arg2}')
    test_data(GenerationData, prompt={'x': '{arg1}\n{conn1} {arg2}', 'y': '{label11}'})