import transformers
from load_model import *

'''
## how to set logging to ignore the warning
# transformers.logging.set_verbosity_debug()
# transformers.logging.set_verbosity_info()
# transformers.logging.set_verbosity_warning()
transformers.logging.set_verbosity_error()

## difference between AutoModel and specific model
model, loading_info = transformers.AutoModel.from_pretrained('roberta-base', output_loading_info=True)
print(model, loading_info)
model, loading_info = transformers.AutoModelForSequenceClassification.from_pretrained('roberta-base', output_loading_info=True)
print(model, loading_info)
# model, loading_info = transformers.RobertaForMaskedLM.from_pretrained('roberta-base', output_loading_info=True)
# print(model, loading_info)

# model, loading_info = transformers.BertForPreTraining.from_pretrained('bert-base-uncased', output_loading_info=True)
# print(model, loading_info)
'''

custom_from_pretrained(
    transformers.AutoModel,
    'roberta-base'
)
custom_from_pretrained(
    transformers.RobertaForMaskedLM,
    'roberta-base'
)