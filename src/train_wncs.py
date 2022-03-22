import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset,concatenate_datasets
import transformers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from transformers import AutoTokenizer, DataCollatorWithPadding,AutoModelForSequenceClassification,AdamW,get_scheduler,TrainingArguments,Trainer,EarlyStoppingCallback
from sklearn.model_selection import ParameterGrid
import logging

def preprocess_data(data,tokenizer,col_name:str):
    """tokenization and necessary processing

    Args:
        data (_type_): _description_
        tokenizer (_type_): _description_
        col_name (str): _description_

    Returns:
        _type_: _description_
    """
    tokenize = lambda data : tokenizer(data[col_name], truncation=True,max_length=512)
    
    data = data.map(tokenize,batched=True)
    data = data.remove_columns([col_name])
    data.set_format("torch")
    return data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


model_name = 'ufal/robeczech-base'
WNC_MODEL_PATH = '/home/horyctom/bias-detection-thesis/src/models/trained/wnc_larger_cs_pretrained.pth'

training_args = TrainingArguments(
            output_dir = './',
            num_train_epochs=3,
            save_total_limit=2,
            disable_tqdm=False,
            per_device_train_batch_size=32,  
            warmup_steps=0,
            weight_decay=0.1,
            logging_dir='./',
            learning_rate=2e-5)

BATCH_SIZE = 32
transformers.utils.logging.set_verbosity_error()

logging.disable(logging.ERROR)


#Prep data
data_wnc = load_dataset('csv',data_files = '/home/horyctom/bias-detection-thesis/data/CS/processed/WNC/wnc.csv')['train']
data_wnc = data_wnc.train_test_split(0.05)
data_wnc['test'].to_csv('/home/horyctom/bias-detection-thesis/data/CS/processed/WNC/wnc_test.csv.',index=False)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,padding=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
wnc_tok = preprocess_data(data_wnc['train'],tokenizer,'sentence')

#Train

model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2);
model.to(device)
trainer = Trainer(model,training_args,train_dataset=wnc_tok,data_collator=data_collator,tokenizer=tokenizer)
trainer.train()
torch.save(model.state_dict(),WNC_MODEL_PATH)