from ..core_model import *


class BaselineClassificationConfig(CustomModelConfig):
    def __init__(
        self, 
        base_model_path=None,
        num_labels=4,
        loss_type='CELoss',
    ) -> None:
        self.base_model_path = base_model_path
        self.num_labels = num_labels
        self.loss_type = loss_type
        
        
class BaselineClassificationModel(CustomModel):
    # predict_with_generate=False
    
    def __init__(
        self, 
        model_config:BaselineClassificationConfig,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        
        self.model:PreTrainedModel = None
        self.initial_model()
        
        if self.model_config.loss_type.lower() == 'celoss':
            self.loss_fn = CELoss()
        else:
            raise Exception('wrong loss_type')
    
    def initial_model(self):
        self.model = custom_from_pretrained(
            AutoModelForSequenceClassification,
            self.model_config.base_model_path,
            num_labels=self.model_config.num_labels,
        )
        self.model_config.transformers_config = AutoConfig.from_pretrained(
            self.model_config.base_model_path, 
            num_labels=self.model_config.num_labels,
        )
    
    def forward(self, input_ids, attention_mask, labels):
        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = model_outputs.logits
        pred = torch.softmax(logits, dim=-1)
        loss = self.loss_fn(pred, labels)
        
        return {
            'logits': logits,
            # 'pred': pred,
            'gt': labels,
            'loss': loss,
        }        
    
    def generate(self, input_ids, attention_mask, labels, *args, **kwargs):
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        pred = torch.argmax(outputs['logits'], dim=1)
        pred = torch.eye(self.model_config.num_labels, dtype=torch.long, device=input_ids.device)[pred]
        return pred
