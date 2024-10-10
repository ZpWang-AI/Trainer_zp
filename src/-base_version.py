from IDRR_data import *

import os
from pathlib import Path as path

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import transformers
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from transformers import (Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer)
from transformers import TrainerCallback, TrainerState, TrainerControl



SRC_DIR = path(__file__).parent
ROOT_DIR = SRC_DIR.parent


# === data ===
dfs = IDRRDataFrames(
    data_name='pdtb2',
    data_level='top',
    data_relation='Implicit',
    data_path='/home/zpwang/IDRR_data/data/used/pdtb2.p1.csv',
)
label_list = dfs.label_list

# === model ===
model_name_or_path = '/home/zpwang/pretrained_models/roberta-base'
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=len(label_list))
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# === args ===
training_args = TrainingArguments(
    output_dir=ROOT_DIR/'output_dir',
    overwrite_output_dir=True,
    
    # strategies of evaluation, logging, save
    evaluation_strategy = "steps", 
    eval_steps = 1,
    logging_strategy = 'steps',
    logging_steps = 1,
    save_strategy = 'steps',
    save_steps = 1,
    max_steps=2,
    
    # optimizer and lr_scheduler
    optim = 'adamw_torch',
    learning_rate = 5e-5,
    weight_decay = 0.01,
    lr_scheduler_type = 'linear',
    warmup_ratio = 0.05,
    
    # epochs and batches 
    num_train_epochs = 10, 
    # max_steps = args.max_steps,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 1,
    
    # train consumption
    eval_accumulation_steps=10,
    bf16=True,
    fp16=False,
)

# === dataset ===
class CustomDataset(Dataset):
    def __init__(self, df, label_list, tokenizer) -> None:
        self.df:pd.DataFrame = df
        label_num = len(label_list)
        self.ys = np.eye(label_num, label_num)[self.df['label11id']]
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        model_inputs = self.tokenizer(
            row['arg1'], row['arg2'],
            add_special_tokens=True, 
            padding=True,
            truncation='longest_first', 
            max_length=512,
        )
        model_inputs['labels'] = self.ys[index]
        return model_inputs
    
    def __len__(self):
        return self.df.shape[0]

train_dataset = CustomDataset(dfs.train_df, label_list, tokenizer)
dev_dataset = CustomDataset(dfs.dev_df, label_list, tokenizer)
test_dataset = CustomDataset(dfs.test_df, label_list, tokenizer)

# === metric ===
class ComputeMetrics:
    def __init__(self, label_list:list) -> None:
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.metric_names = ['Macro-F1', 'Acc']
    
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
        
        pred = pred[..., :len(self.label_list)]
        labels = labels[..., :len(self.label_list)]
        
        # pred = pred!=0
        max_indices = np.argmax(pred, axis=1)
        bpred = np.zeros_like(pred, dtype=int)
        bpred[np.arange(pred.shape[0]), max_indices] = 1
        pred = bpred
        assert ( pred.sum(axis=1)<=1 ).sum() == pred.shape[0]
        labels = labels!=0
        
        res = {
            'Macro-F1': f1_score(labels, pred, average='macro', zero_division=0),
            'Acc': np.sum(pred*labels)/len(pred),
        }
        return res

# === callback ===
class CustomCallback(TrainerCallback):
    def __init__(
        self, 
        log_filepath=None,
    ):
        super().__init__()
        if log_filepath:
            self.log_filepath = log_filepath
        else:
            self.log_filepath = ROOT_DIR / 'output_dir' / 'log.jsonl'
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.log_filepath, 'a', encoding='utf8')as f:
            f.write(str(kwargs['logs'])+'\n')

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        pass
    

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=ComputeMetrics(dfs.label_list),
    callbacks=[CustomCallback()],
)

train_result = trainer.train()
test_result = trainer.evaluate(eval_dataset=test_dataset)
print(f'> train_result:\n  {train_result}')
print(f'> test_result:\n  {test_result}')