import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from corpy.morphodita import Tokenizer

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_checkpoint = 'ufal/robeczech-base'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
transformers.logging.set_verbosity(transformers.logging.ERROR)

def classify_sentence(sent:str):
    toksentence = tokenizer(sent,truncation=True,return_tensors="pt")
    model.eval()
    with torch.no_grad():
        toksentence.to(device)
        output = model(**toksentence)
    
    return F.softmax(output.logits,dim=1).argmax(dim=1)
    
def classify_text(text:str):
    tokenizer_morphodita = Tokenizer("czech")

    all = []
    for sentence in tokenizer_morphodita.tokenize(text, sents=True):
        all.append(sentence)

    sentences = np.array([' '.join(x) for x in all])
    annotations = np.array(list(map(classify_sentence,sentences)))
    
    return annotations
    
def classify_text_wrapper(text:str):
    result = classify_text(text)
    n = len(result)
    non_biased = np.where(result==0)[0].shape[0]
    biased = np.where(result==1)[0].shape[0]

    return {'Non-biased':non_biased/n,'Biased':biased/n}
    
    
def interpret_bias(text:str):
    result = classify_text(text)
    
    tokenizer_morphodita = Tokenizer("czech")

    interpretation = []
    all = []
    for sentence in tokenizer_morphodita.tokenize(text, sents=True):
        all.append(sentence)
    
    sentences = np.array([' '.join(x) for x in all])
    
    for idx,sentence in enumerate(sentences):
        score = 0
        #non biased
        if result[idx] == 0:
            score = -1
        #biased
        if result[idx] == 1:
            score = 1
        interpretation.append((sentence, score))
        
    return interpretation
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained("sagittariusA/media_bias_classifier_cs")
model.eval()

label = gr.outputs.Label(num_top_classes=2)
inputs = gr.inputs.Textbox(placeholder=None, default="", label=None)
app = gr.Interface(fn=classify_text_wrapper,title='Bias classifier',theme='default',
                    inputs="textbox",layout='unaligned', outputs=label, capture_session=True
                    ,interpretation=interpret_bias)

app.launch(inbrowser=True)

