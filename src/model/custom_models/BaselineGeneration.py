from ..core_model import *


class BaselineGenerationConfig(AttrDict):
    def __init__(
        self, 
        base_model_path=None,
        generation_config:dict={},
        label_list=None,
        # loss_type='CELoss',
    ) -> None:
        self.base_model_path = base_model_path
        self.generation_config = generation_config
        self.label_list = label_list
        # self.loss_type = loss_type
        
        
class BaselineGenerationModel(CustomModel):
    predict_with_generate=True
    
    def __init__(
        self, 
        base_model_path,
        generation_config,
        label_list,
        # loss_type,
    ) -> None:
        super().__init__()
        
        self.base_model_path = base_model_path
        self.label_list = label_list
        self.num_labels = len(label_list)
        
        self.model:PreTrainedModel = None
        self.model_config = None
        self.tokenizer = None
        self.initial_model()
        
        self.generation_config = GenerationConfig(**generation_config)
        self.vote_num = self.generation_config.num_return_sequences
    
    def initial_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_path, 
        )
        self.model_config = AutoConfig.from_pretrained(
            self.base_model_path, 
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
    
    def forward(self, input_ids, attention_mask, labels, *args, **kwargs):
        # if self.training:
        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # print(input_ids.shape, labels.shape)
        # outs = format_output(dict(model_outputs))
        # print(outs)
        # exit()
        logits = model_outputs.logits
        # pred = torch.softmax(logits, dim=-1)
        # loss = self.loss_fn(pred, labels)
        # generate_outputs = self.model.generate(input_ids, generation_config=self.generation_config)
        
        output = {
            'logits': logits,
            # 'pred': pred,
            'gt': labels,
            'loss': model_outputs['loss']
        }      
        return output
    
    def vote_pred(self, pred):
        def vote_piece(piece):
            dic = Counter(piece)
            if self.num_labels in dic:
                del dic[self.num_labels]
            
            if len(dic):
                return max(dic, key=lambda x:dic[x])
            else:
                return self.num_labels
        
        return [
            vote_piece(pred[pid:pid+self.vote_num])
            for pid in range(0, len(pred), self.vote_num)
        ]
        
    def postprocess_pred(self, pred_sentence:str):
        # get the first occuring label with longest length
        res_lid = self.num_labels
        res_index = 10**9
        for lid, label in enumerate(self.label_list):
            if label in pred_sentence:
                index = pred_sentence.index(label)-len(label)
                if index < res_index:
                    res_index = index
                    res_lid = lid
        return res_lid
    
    def generate(self, input_ids, attention_mask, labels, *args, **kwargs):
        model_outputs = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            generation_config=self.generation_config
        )
        pred = self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
        pred = [self.postprocess_pred(p)for p in pred]
        
        if self.vote_num > 1:
            pred = self.vote_pred(pred)

        # pred = torch.tensor(pred, dtype=torch.long, device=input_ids.device)
        pred = torch.eye(self.num_labels+1, self.num_labels, dtype=torch.long, device=input_ids.device)[pred]
        return pred