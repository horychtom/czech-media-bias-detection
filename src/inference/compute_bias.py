import torch
import torch.nn.functional as F
import transformers

from transformers import AutoTokenizer, DataCollatorWithPadding

from src.utils.myutils import *
from tqdm import tqdm
import logging
import warnings
import re
import pandas as pd
from nltk import sent_tokenize

logging.disable(logging.ERROR)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning) 


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = 'fav-kky/FERNET-C5'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,padding=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("horychtom/czech_media_bias_classifier")

BATCH_SIZE = 32


def classify_sentence(sent:str):
    toksentence = tokenizer(sent,truncation=True,return_tensors="pt",max_length=128)
    model.eval()
    model.to(device)
    with torch.no_grad():
        toksentence.to(device)
        output = model(**toksentence)
    
    classification = F.softmax(output.logits,dim=1).argmax(dim=1)
    
    return classification[0].item()

def strip_domains(data):
    get_domain = lambda x : '.'.join(x.split('.')[-2:])
    get_section = lambda x : x.split('.')[0] if '.' in x else x
    data['subdomain'] = data['subdomain'].apply(get_domain)
    data['section'] = data['section'].apply(get_section)
    
    return data

def date_format(data):
    data['published'] = data['published'].apply(lambda x: x[:7])
    data = data.assign(Year = data['published'].apply(lambda x: int(x[:4])))
    
    return data

def assign_bias_ratio(data,split:str):
    size = len(data)
    values = np.zeros(size)
    
    for i in tqdm(range(size)):
        text = data.iloc[i][split]
        sentences = sent_tokenize(text)
        labels = np.array(list(map(classify_sentence,sentences)))
        values[i] = 100*np.sum(labels)/len(labels)
    
    data.insert(0,split+'_bias',values)
    
    return data

def assign_quote_ratio(data):
    size = len(data)
    qvalues = np.zeros(size)

    r = re.compile('„|"')
    
    for i in tqdm(range(size)):
        text = data.iloc[i]['text']
        sentences = sent_tokenize(text)
        
        # quoting
        result = np.array(list(map(r.search, sentences)))
        qindices = np.where(np.array(result) != None)[0]

        qvalues[i] = 100*len(qindices)/len(sentences)
    
    data.insert(0,'quoting_ratio',qvalues)
    
    return data

def assign_bias_headline(data):
    size = len(data)
    values = np.zeros(size)
    
    for i in tqdm(range(size)):
        text = data.iloc[i]['headline']
        values[i] = classify_sentence(text)
        
    data.insert(0,'headline_bias',values)
    
    return data


# DATA
test = pd.read_json('/mnt/data/factcheck/summarization/sumeczech/sumeczech-1.0-test.jsonl',lines=True)
dev = pd.read_json('/mnt/data/factcheck/summarization/sumeczech/sumeczech-1.0-dev.jsonl',lines=True)
data = pd.concat([test,dev])

#preprocess
data = data[~data['subdomain'].str.contains('blog')]
data = data[data['published']!='']
data = date_format(data)
data = strip_domains(data)
data.drop(['filename', 'dataset','md5','offset'], axis=1, inplace=True)

idnes = data[data.subdomain == 'idnes.cz']

idnes = assign_quote_ratio(idnes)

idnes.to_csv('/home/horyctom/bias-detection-thesis/notebooks/media/full_pipeline/idnes.csv',index=False)

idnes = assign_bias_headline(idnes)

idnes.to_csv('/home/horyctom/bias-detection-thesis/notebooks/media/full_pipeline/idnes.csv',index=False)

idnes = assign_bias_ratio(idnes,'abstract')

idnes.to_csv('/home/horyctom/bias-detection-thesis/notebooks/media/full_pipeline/idnes.csv',index=False)

idnes = assign_bias_ratio(idnes,'text')

idnes.to_csv('/home/horyctom/bias-detection-thesis/notebooks/media/full_pipeline/idnes.csv',index=False)
