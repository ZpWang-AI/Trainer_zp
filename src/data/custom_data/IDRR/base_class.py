from ...core_data import *

with ignore_exception:
    from IDRR_data import IDRRDataFrames, PromptFiller


class IDRRComputeMetrics(CustomComputeMetrics):
    label_list: List[str]
    metric_names: List[str]
    
    def __init__(self, label_list:list) -> None:
        self.label_list = label_list
        self.label_num = len(label_list)
        # self.metric_names = ['Macro-F1', 'Acc']+label_list
        self.metric_names = ['Macro-F1']
    
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
        
        pred = pred[..., :self.label_num]
        labels = labels[..., :self.label_num]
        
        pred = pred!=0
        # assert ( pred.sum(axis=1)<=1 ).sum() == pred.shape[0]
        # # pred only one data_relation or no data_relation
        labels = labels!=0
        
        res = {
            'Macro-F1': f1_score(labels, pred, average='macro', zero_division=0),
            # 'Acc': np.sum(pred*labels)/len(pred),
        }
        
        # for i, target_type in enumerate(self.label_list):
        #     res[target_type] = f1_score(pred[:,i], labels[:,i], zero_division=0)
        
        return res
    

class IDRRDataCollator(CustomDataCollator):
    # tokenizer
    def __init__(self, tokenizer):
        raise Exception()
    #     self.tokenizer = tokenizer
        
    def __call__(self, features, return_tensors=None):
        pass
        raise Exception()
    

class IDRRDataset(CustomDataset):
    def __init__(
        self, 
        tokenizer:transformers.PreTrainedTokenizer,
        x_strs:List[str], 
        y_strs:List[str]=None,
        y_nums:List[List[float]]=None,
        max_length:int=512,
        shift_num:int=0,
        # **kwargs
    ) -> None:
        super().__init__()
        
        self.tokenizer = tokenizer
        self.x_strs = x_strs
        self.y_strs = [self.token_encode(p).input_ids for p in y_strs] if y_strs is not None else None
        self.y_nums = y_nums
        self.max_length = max_length    
        # assert max_length <= tokenizer.max_model_input_sizes
        
        self.n = len(self.x_strs)
        self.shift_num = shift_num  # manually random the data by shiftting indices
        
    def token_encode(self, target_str:str):
        if '<sep>' in target_str:
            _input = target_str.split('<sep>')
        else:
            _input = [target_str]
        assert len(_input) <= 2
        _input = [p.strip() for p in _input]
        
        return self.tokenizer(
            *_input,
            add_special_tokens=True, 
            padding=True,
            truncation='longest_first', 
            max_length=self.max_length,
        )
    
    def __getitem__(self, index):
        assert 0 <= index < self.n
        index = (index+self.shift_num) % self.n
        model_inputs = self.token_encode(self.x_strs[index])
        
        if self.y_strs is not None and self.y_nums is not None:
            model_inputs['labels'] = {
                'str': self.y_strs[index],
                'num': self.y_nums[index],
            }
    
        elif self.y_strs is not None:
            model_inputs['labels'] = self.y_strs[index]
    
        elif self.y_nums is not None:
            model_inputs['labels'] = self.y_nums[index]
        
        # for key in self.extra_kwargs:
        #     model_inputs[key] = self.extra_kwargs[key][index]
    
        return model_inputs
    
    def __len__(self):
        return self.n


class IDRRDataConfig(CustomDataConfig):
    def __init__(
        self,
        data_path,
        data_name='pdtb2',
        data_level='top',
        data_relation='Implicit',
        
        base_model_path='roberta-base',
        prompt={'x':'{arg1} {arg2}'},
        
        max_length=1024,
        secondary_label_weight=0.5,
        mini_dataset=False,
    ):
        super().__init__()
        
        self.data_path = data_path
        self.data_name = data_name
        self.data_level = data_level
        self.data_relation = data_relation
        
        self.base_model_path = base_model_path
        self.prompt = prompt
        if 'x' in prompt:
            self.prompt['train_x'] = prompt['x']
            self.prompt['eval_x'] = prompt['x']
        if 'y' in prompt:
            self.prompt['train_y'] = prompt['y']
            self.prompt['eval_y'] = prompt['y']
        
        self.max_length = max_length
        self.secondary_label_weight = secondary_label_weight
        self.mini_dataset = mini_dataset


class IDRRData(CustomData):
    test_x_ignore=(
        # 'reason', 
        # 'conn1', 'conn2', 'conn1id', 'conn2id', 
        # 'conn1sense1', 'conn1sense2', 'conn2sense1', 'conn2sense2',
        # 'conn1sense1id', 'conn1sense2id', 'conn2sense1id', 'conn2sense2id',
    )
    train_input_y_nums:bool
    train_input_y_strs:bool
    
    def __init__(
        self, 
        data_config:IDRRDataConfig,
    ):
        self.data_config = data_config
        _config = data_config
        self.dataframes = IDRRDataFrames(
            data_name=_config.data_name,
            data_level=_config.data_level,
            data_relation=_config.data_relation,
            data_path=_config.data_path,
        )
        
        # self.base_model_path = base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(_config.base_model_path)
        # self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        # self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.prompt = _config.prompt

        assert 'train_x' in self.prompt and 'eval_x' in self.prompt
        assert all(ignore_key not in self.prompt['eval_x']
                   for ignore_key in self.test_x_ignore)

        self.label_list = self.dataframes.label_list
        self.num_labels = len(self.label_list)
        self.label_map = {label:p for p, label in enumerate(self.label_list)}

        self.data_collator:CustomDataCollator = self.get_data_collator()
        self.compute_metrics:CustomComputeMetrics = self.get_compute_metrics()
    
    def get_data_collator(self):
        raise Exception()
    
    def get_compute_metrics(self):
        # raise Exception()
        return CustomComputeMetrics(self.label_list)
    
    def get_label_vector(
        self, 
        df:pd.DataFrame,
        # label_ids:pd.Series,
        # secondary_label_ids:List[pd.Series]=None,
        secondary_label_weight=0.,
    ):
        eye = np.eye(self.num_labels+1, self.num_labels)
        primary_label_ids = df['label11id'].astype(int)
        label_vector = eye[primary_label_ids]
        if secondary_label_weight:
            eye *= secondary_label_weight
            for sec_label_ids in [df['label12id'], df['label21id'], df['label22id']]:
                sec_label_ids = sec_label_ids.copy()
                sec_label_ids[pd.isna(sec_label_ids)] = self.num_labels
                sec_label_ids = sec_label_ids.astype(int)
                label_vector += eye[sec_label_ids]
        return label_vector
    
    def get_preprocessed_dataframe(self, split:Literal['train', 'dev', 'test',]) -> pd.DataFrame:
        df = self.dataframes.get_dataframe(split)
        
        if 'subtext_res' in df.columns:
            if self.subtext_threshold == 1:
                df.loc[df['subtext_res']!=1, 'subtext'] = ''
            elif self.subtext_threshold == 2:
                df.loc[df['subtext_res']==0, 'subtext'] = ''
            else:
                pass        
        
        if split != 'train':
            if self.mini_dataset:
                df = df.iloc[:4]
            return df
        
        else:
            if self.mini_dataset:
                df = df.iloc[:8]
            
            # # flatten all sense
            # if self.data_augmentation_flatten_sec_label:
            #     df2 = df.copy()
            #     df2.dropna(subset=['conn1sense2'], inplace=True)
            #     df2['conn1sense1'] = df2['conn1sense2']
            #     df3 = df.copy()
            #     df3.dropna(subset=['conn2sense1'], inplace=True)
            #     df3['conn1sense1'], df3['conn1'] = df3['conn2sense1'], df3['conn2']
            #     df4 = df.copy()
            #     df4.dropna(subset=['conn2sense2'], inplace=True)
            #     df4['conn1sense1'], df4['conn1'] = df4['conn2sense2'], df4['conn2']
            #     df = pd.concat([df,df2,df3,df4], ignore_index=True)
            #     for pdna in 'conn2 conn1sense2 conn2sense1 conn2sense2'.split():
            #         df[pdna] = pd.NA
            
            # # concat connective with arg2, flatten and clear, return [init, conn1+arg2, conn2+arg2]
            # if self.data_augmentation_add_conn_to_arg2:
            #     df2 = df.copy()
            #     df2['arg2'] = df2['conn1']+df2['arg2']
            #     df3 = df.copy()
            #     df3.dropna(subset=['conn2'], inplace=True)
            #     df3['arg2'] = df3['conn2']+df3['arg2']
            #     df3['conn1'] = df3['conn2']
            #     df3['conn1sense1'] = df3['conn2sense1']
            #     df3['conn1sense2'] = df3['conn2sense2']
            #     df = pd.concat([df, df2, df3], ignore_index=True)
            #     for pdna in 'conn2 conn2sense1 conn2sense2'.split():
            #         df[pdna] = pd.NA
            
            # if (self.data_augmentation_flatten_sec_label or 
            #     self.data_augmentation_add_conn_to_arg2):
            #     df = self.dataframes.process_df_sense(df)
                
            return df
    
    def get_dataset(
        self,
        split:Literal['train', 'dev', 'test'],
    ) -> CustomDataset:
        df = self.get_preprocessed_dataframe(split=split)
        if split == 'train':
            return CustomDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(df=df, prompt=self.prompt['train_x'],
                                         tokenizer=self.tokenizer)),
                y_nums=(
                    self.get_label_vector(
                        df=df,
                        secondary_label_weight=self.secondary_label_weight,
                    )
                    if self.train_input_y_nums else None
                ),
                y_strs=(
                    list(PromptFiller(df=df, prompt=self.prompt['train_y'],
                                      tokenizer=self.tokenizer))
                    if self.train_input_y_strs else None
                ),
                max_length=self.max_length,
                shift_num=self.shift_num,
            )
        else:
            return CustomDataset(
                tokenizer=self.tokenizer,
                x_strs=list(PromptFiller(
                    df=df, prompt=self.prompt['eval_x'], 
                    ignore=self.test_x_ignore,
                    tokenizer=self.tokenizer,
                )),
                y_nums=self.get_label_vector(
                    df=df,
                    secondary_label_weight=1,
                ),
                max_length=self.max_length,
            )
            
    @property
    def train_dataset(self):
        return self.get_dataset(split='train')
    
    @property
    def dev_dataset(self):
        return self.get_dataset(split='dev')

    @property
    def test_dataset(self):
        return self.get_dataset(split='test')