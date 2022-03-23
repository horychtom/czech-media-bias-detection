# Making imports convenient
import sys
import os
PATH='/home/horyctom/bias-detection-thesis'
sys.path.insert(1, PATH)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset,concatenate_datasets
import transformers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from transformers import AutoTokenizer, DataCollatorWithPadding,AutoModelForSequenceClassification,AdamW,get_scheduler,TrainingArguments,Trainer,EarlyStoppingCallback
from sklearn.model_selection import ParameterGrid
from src.utils.myutils import *
import yaml
import json
import logging

logging.disable(logging.ERROR)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


model_name = 'ufal/robeczech-base'
CONFIG_PATH = PATH + '/src/utils/config.yaml'
WNC_MODEL_PATH = '/home/horyctom/bias-detection-thesis/src/models/trained/wnc_larger_cs_pretrained.pth'

training_args = TrainingArguments(
            output_dir = './',
            num_train_epochs=3,
            save_total_limit=2,
            disable_tqdm=False,
            per_device_train_batch_size=16,  
            warmup_steps=0,
            weight_decay=0.1,
            logging_dir='./',
            learning_rate=2e-5)

BATCH_SIZE = 16
from tqdm import tqdm


babe = load_dataset('csv',data_files = PATH + '/data/CS/processed/BABE/train.csv')['train']
subj = load_dataset('csv',data_files=PATH + '/data/CS/raw/SUBJ/subj.csv')['train']


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,padding=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

babe_tok = preprocess_data(babe,tokenizer,'sentence')
unlabelled_tok = preprocess_data(subj,tokenizer,'sentence')

k=100
scores=[]
for train_index, val_index in skfold.split(babe_tok['input_ids'],babe_tok['label']):

    #split for this whole selftraining iteration
    token_train = Dataset.from_dict(babe_tok[train_index])
    token_valid = Dataset.from_dict(babe_tok[val_index])
    eval_dataloader = DataLoader(token_valid, batch_size=BATCH_SIZE, collate_fn=data_collator)
    unlabelled_tok = preprocess_data(subj,tokenizer,'sentence')

    
    #self training
    while True:
        #print("Iteration :",iterations)
        print("Fitting on ", len(token_train), " data")
        
        #initial training
        #torch.cuda.manual_seed(12345)
        #torch.manual_seed(12345)
        model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2)
        trainer = Trainer(model,training_args,train_dataset=token_train,data_collator=data_collator,tokenizer=tokenizer)
        trainer.train()
        
        #making predictions on unlabelled dataset
        unlabelled_dataloader = DataLoader(unlabelled_tok, batch_size=BATCH_SIZE, collate_fn=data_collator)
        logits = torch.Tensor().to(device)

        #make predictions on unlabelled
        model.eval()
        for batch in unlabelled_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = torch.cat((logits,F.softmax(outputs.logits)))

        
        #stop when there is not enough of resources
        if len(logits[:,0]) < k or len(logits[:,1]) < k:
            break
            
        #only sample 2x size of BABE
        if len(token_train) > 3000:
            break
            
        #indices of the highest probability ranked predictions
        unbiased_topk_indices = torch.topk(logits[:,0],k)[1]
        biased_topk_indices = torch.topk(logits[:,1],k)[1]
        indices = torch.cat((unbiased_topk_indices,biased_topk_indices)).cpu()

        #create new augmentation and concat it
        masks = unlabelled_tok[indices]['attention_mask']
        input_ids = unlabelled_tok[indices]['input_ids']
        labels = [0]*len(unbiased_topk_indices) + [1]*len(biased_topk_indices)
        to_add = Dataset.from_dict({'attention_mask':masks,'input_ids':input_ids,'label':labels})
        
        token_train = concatenate_datasets([to_add,token_train]).shuffle(seed=42)

        #remove them from unlabelled
        all_indices = np.arange(0,len(unlabelled_tok))
        remaining = np.delete(all_indices,indices)
        unlabelled_tok = Dataset.from_dict(unlabelled_tok[remaining])
        
        print(compute_metrics(model,device,eval_dataloader)['f1'])
    #evaluation
    scores.append(compute_metrics(model,device,eval_dataloader)['f1'])
    print("FINAL SCORE: ",scores[-1])
    
    
print(scores)
with open('./results.txt','w') as f:
    f.write(str(scores))