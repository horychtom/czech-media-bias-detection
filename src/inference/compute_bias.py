imprt numpy as np
from transformers import AutoModelForSequenceClassification
import torch


#setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.disable(logging.ERROR)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning) 


def classify_sentence(sent:str):
    toksentence = tokenizer(sent,truncation=True,return_tensors="pt",max_length=128)
    model.eval()
    model.to(device)
    with torch.no_grad():
        toksentence.to(device)
        output = model(**toksentence)
    
    classification = F.softmax(output.logits,dim=1).argmax(dim=1)
    
    return classification[0].item()


def classify_sentence_probability(sent:str):
    toksentence = tokenizer(sent,truncation=True,return_tensors="pt",max_length=128)
    model.eval()
    model.to(device)
    with torch.no_grad():
        toksentence.to(device)
        output = model(**toksentence)
    
    classification = F.softmax(output.logits,dim=1).squeeze(dim=0)
    
    return classification

model = AutoModelForSequenceClassification.from_pretrained("horychtom/czech_media_bias_classifier")