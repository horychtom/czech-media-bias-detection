import torch
import torch.nn.functional as F
from datasets import load_dataset, load_metric
import numpy as np

from transformers import AutoTokenizer, DataCollatorWithPadding,AutoModelForSequenceClassification,TrainingArguments,Trainer
import logging

logging.disable(logging.ERROR)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


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

def compute_metrics_eval(eval_preds):
    metric = load_metric("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(average='macro',predictions=predictions, references=labels)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


model_name = 'ufal/robeczech-base'
WNC_MODEL_PATH = '/home/horyctom/bias-detection-thesis/src/models/trained/wncs_pretrained.pth'
BATCH_SIZE = 64


training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,  
    per_device_eval_batch_size=BATCH_SIZE,
    eval_steps=1000,
    logging_steps=1000,
    save_steps=1000,
    disable_tqdm = False,
    warmup_steps=0,
    save_total_limit=5,
    evaluation_strategy="steps",
    load_best_model_at_end = True,
    metric_for_best_model = 'f1',
    weight_decay=0.2,
    output_dir = './',
    learning_rate=1e-5)


#Prep data
data_wnc = load_dataset('csv',data_files = '/home/horyctom/bias-detection-thesis/data/CS/processed/WNC/wnc.csv')['train']
data_wnc = data_wnc.train_test_split(0.1,seed=42)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,padding=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train = preprocess_data(data_wnc['train'],tokenizer,'sentence')
test = preprocess_data(data_wnc['test'],tokenizer,'sentence')


#Train
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2);
model.to(device)
trainer = Trainer(model,training_args,train_dataset=train,data_collator=data_collator,tokenizer=tokenizer,eval_dataset=test,
                          compute_metrics=compute_metrics_eval)
trainer.train()
torch.save(model.state_dict(),WNC_MODEL_PATH)