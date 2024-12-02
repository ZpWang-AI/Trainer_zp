from ..core_data import *

with ignore_exception:
    from IDRR_data import *


class TESTComputeMetrics(CustomComputeMetrics):
    def __init__(self) -> None:
        self.metric_names = ['f1']
    
    def __call__(self, eval_pred):
        """
        n = label categories
        eval_pred: (pred, labels)
        # pred: np.array [datasize, ]
        pred: np.array [datasize, n]
        labels: np.array [datasize, n]
        X[p][q]=True, sample p belongs to label q (False otherwise)
        """
        pred, labels = eval_pred
        pred: np.ndarray
        labels: np.ndarray
        # print(pred, labels)
        # pred = self.process_pred(pred)
        # pred = np.argmax(pred[0], axis=1)
        # pred = np.eye(self.num_labels+1, self.num_labels)[pred]
        
        pred = pred[..., :len(self.label_list)]
        labels = labels[..., :len(self.label_list)]
        
        pred = pred!=0
        assert ( pred.sum(axis=1)<=1 ).sum() == pred.shape[0]
        # pred only one data_relation or no data_relation
        labels = labels!=0
        # labels = (labels != 0).astype(int)
        # print(pred, labels)
        # exit()
        
        res = {
            'Macro-F1': f1_score(labels, pred, average='macro', zero_division=0),
            # 'Acc': np.sum(pred*labels)/len(pred),
        }
        
        # for i, target_type in enumerate(self.label_list):
        #     res[target_type] = f1_score(pred[:,i], labels[:,i], zero_division=0)
        
        return res
    

class TESTDataCollator:
    # tokenizer
    def __init__(self, tokenizer):
        raise Exception()
    #     self.tokenizer = tokenizer
        
    def __call__(self, features, return_tensors=None):
        raise Exception()
    

class TESTDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        raise Exception()
    
    def __len__(self):
        raise Exception()


class TESTDataConfig(AttrDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.trainset_size = None
        self.devset_size = None
        self.testset_size = None
        
    def refill(self, data_:"TESTData"):
        for split in 'train dev test'.split():
            self[f'{split}set_size'] = len(data_.get_dataset(split))
            

class TESTData:
    def __init__(self, data_config:TESTDataConfig=None) -> None:
        self.data_config = data_config

        self.data_collator:TESTDataCollator = self.get_data_collator()
        self.compute_metrics:TESTComputeMetrics = self.get_compute_metrics()
    
    def get_data_collator(self):
        raise Exception()
    
    def get_compute_metrics(self):
        raise Exception()
    
    def get_dataset(
        self,
        split:Literal['train', 'dev', 'test'],
    ) -> TESTDataset:
        raise Exception()


class TestDataset(TESTDataset):
    pass
    def __init__(
        self, 
        tokenizer,
        x_strs:List[str], 
        y_nums:List[Union[float, List[float]]]=None,
        y_strs=None,
        # **kwargs
    ) -> None:
        super().__init__(tokenizer=tokenizer, x_strs=x_strs)
        
        self.tokenizer = tokenizer
        self.x_strs = x_strs
        # self.y_nums = y_nums
        
    def __getitem__(self, index):
        model_inputs = self.tokenizer(
            self.x_strs[index],
            add_special_tokens=True, 
            padding=True,
            truncation='longest_first', 
            max_length=1024,
        )
        # model_inputs['labels'] = self.tokenizer(
        #     self.y_strs[index],
        #     add_special_tokens=True, 
        #     padding=True,
        #     truncation='longest_first', 
        #     max_length=256,
        # ).input_ids
        model_inputs['labels'] = [index]
        
        # for key in self.extra_kwargs:
        #     model_inputs[key] = self.extra_kwargs[key][index]
    
        return model_inputs
    
    def __len__(self):
        return len(self.x_strs)


class TestData(TESTData):
    train_input_y_nums = True
    train_input_y_strs = False
    
    def get_data_collator(self):
        return DataCollatorForSeq2Seq(self.tokenizer)
    
    def get_dataset(
        self,
        split:Literal['train', 'dev', 'test', 'blind_test'],
    ) -> TESTDataset:
        df = self.get_preprocessed_dataframe(split=split)
        if split == 'train':
            return TestDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(df=df, prompt=self.prompt['x'])),
            )
        else:
            return TestDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(
                    df=df, prompt=self.prompt['x'], 
                    ignore=self.test_x_ignore
                )),
            )