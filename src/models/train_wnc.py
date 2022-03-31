# Making imports convenient
import sys
import os
PATH=os.getcwd().split('/src')[0] + "/bias-detection-thesis/"
sys.path.insert(1,PATH)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_metric,load_dataset,Dataset

import transformers
from transformers import AutoTokenizer, DataCollatorWithPadding,RobertaForSequenceClassification,AdamW,get_scheduler,TrainingArguments,Trainer

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold
from tqdm.auto import tqdm, trange

from src.utils.myutils import clean_memory,compute_metrics,preprocess_data

model_checkpoint = 'roberta-base'

def compute_metrics(eval_preds):
    metric = load_metric("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
transformers.utils.logging.set_verbosity_error()
BATCH_SIZE = 32


wnc = load_dataset('csv',data_files=PATH+"/data/EN/processed/WNC/wnc.csv")['train'].train_test_split(test_size=0.1)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint);
model = RobertaForSequenceClassification.from_pretrained(model_checkpoint);
model.to(device);

train_tokenized = preprocess_data(wnc['train'],tokenizer,'sentence')
val_tokenized = preprocess_data(wnc['test'],tokenizer,'sentence')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=BATCH_SIZE,  
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=1000,
    save_steps=1000,
    disable_tqdm = False,
    warmup_steps=2000,
    save_total_limit=2,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    metric_for_best_model = 'f1',
    weight_decay=0.1,
    output_dir = './',
    learning_rate=2e-5)

model = RobertaForSequenceClassification.from_pretrained(model_checkpoint);
trainer = Trainer(model,training_args,train_dataset=train_tokenized,eval_dataset=val_tokenized,compute_metrics=compute_metrics,data_collator=data_collator,
                      tokenizer=tokenizer)
trainer.train()

torch.save(model.state_dict(),PATH + "/src/models/wnc.pth")