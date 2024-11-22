from ..core_model import *
from .PCPModel import PCPModel


class CPKDConfig(AttrDict):
    def __init__(
        self, 
        base_model_path=None,
        teacher_ckpt_path=None,
        label_list:list=None,
        ans_word_list_token:List[int]=None,
        ans_label_list_id:List[int]=None,
        temperature_rate:float=1,
        loss_type='CELoss',
        loss_weight_distil:float=0.4,
    ) -> None:
        self.base_model_path = base_model_path
        self.teacher_ckpt_path = teacher_ckpt_path
        self.label_list = label_list
        self.ans_word_list_token = ans_word_list_token
        self.ans_label_list_id = ans_label_list_id
        self.temperature_rate = temperature_rate
        self.loss_type = loss_type
        self.loss_weight_distil = loss_weight_distil
        
        
class CPKDModel(CustomModel):
    def __init__(
        self, 
        base_model_path,
        teacher_ckpt_path,
        label_list,
        ans_word_list_token:List[int],
        ans_label_list_id:List[int],
        temperature_rate:float,
        loss_type,
        loss_weight_distil,
    ) -> None:
        super().__init__()
        
        self.base_model_path = base_model_path
        self.teacher_ckpt_path = teacher_ckpt_path
        self.label_list = label_list
        self.num_labels = len(label_list)
        
        self.teacher_model:PCPModel = None
        self.model:nn.Module = None
        self.model_config = None
        self.tokenizer:transformers.PreTrainedTokenizer = None
        self.generation_config = GenerationConfig()
        
        self.ans_word_list_token = ans_word_list_token
        self.ans_label_list_id = ans_label_list_id
        
        self.temperature_rate = temperature_rate
        self.loss_type = loss_type
        self.loss_weight_distil = loss_weight_distil
        if loss_type.lower() == 'celoss':
            self.loss_fn = CELoss()
        else:
            raise Exception('wrong loss_type')
        self.loss_fn_distil = KLDivLoss()
        
        self.initial_model()
    
    def initial_model(self):
        self.teacher_model = PCPModel(
            base_model_path=self.base_model_path,
            label_list=self.label_list,
            ans_word_list_token=self.ans_word_list_token,
            ans_label_list_id=self.ans_label_list_id,
            loss_type=self.loss_type,
        )
        self.teacher_model.load_state_dict(torch.load(self.teacher_ckpt_path))
        self.teacher_model.eval()
        self.model = transformers.RobertaForMaskedLM.from_pretrained(
            self.base_model_path, 
        )
        self.model_config = AutoConfig.from_pretrained(
            self.base_model_path, 
        )
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(
            self.base_model_path,
        )
    
    def forward(
        self,
        input_ids:torch.Tensor, 
        attention_mask, 
        teacher_input_ids:torch.Tensor,
        teacher_attention_mask,
        labels
    ):
        if labels.shape[1] == self.num_labels:
            return {'loss': torch.tensor(-1.0)}
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = model_outputs.logits  # bs, seq_len, vocab_size
        
        mask_positions = torch.where(input_ids == self.tokenizer.mask_token_id)
        mask_logits = logits[mask_positions]  # bs, vocab_size
        ans_word_logits = mask_logits[:, self.ans_word_list_token]  # bs, ans_word_num
        
        ans_word_prob = torch.softmax(
            ans_word_logits/self.temperature_rate, dim=-1)  # bs
        loss_ce = self.loss_fn(ans_word_prob, labels)

        with torch.no_grad():
            teacher_forward_outputs = self.teacher_model.forward(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                labels=ans_word_prob,
            )
            teacher_prob = torch.softmax(
                teacher_forward_outputs['logits']/self.temperature_rate, dim=-1)
        loss_distil = self.loss_fn_distil(ans_word_prob, teacher_prob)
        loss = (1-self.loss_weight_distil)*loss_ce + \
               self.loss_weight_distil*(self.temperature_rate**2)*loss_distil
        
        return {
            'logits': ans_word_logits,
            'gt': labels,
            'loss': loss,
        }
    
    def generate(self, input_ids, attention_mask, labels, *args, **kwargs):
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = model_outputs.logits  # bs, seq_len, vocab_size
        
        mask_positions = torch.where(input_ids == self.tokenizer.mask_token_id)
        mask_logits = logits[mask_positions]  # bs, vocab_size
        ans_word_logits = mask_logits[:, self.ans_word_list_token]  # bs, ans_word_num
        
        ans_word_pred = torch.argmax(ans_word_logits, dim=-1)  # bs
        ans_label_pred = [self.ans_label_list_id[p]for p in ans_word_pred]
        pred_vec = torch.eye(self.num_labels, dtype=torch.long, 
                             device=input_ids.device)[ans_label_pred]
        return pred_vec
        