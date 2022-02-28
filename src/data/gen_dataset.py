from datasets import load_dataset,Dataset

import argparse
from SUBJ import SUBJ
from DataCreator import DataCreator
from MPQA import MPQA

parser = argparse.ArgumentParser(description='Ukraine transform script.')
parser.add_argument("-t","--translated", help="Toggle if you already have translated data", action="store_true")
parser.add_argument("-d","--dataset", help="dataset name",required=True)

args = parser.parse_args()

dataset = args.dataset

if dataset == "UA-crisis":
    dc = DataCreator(dataset,"sentences-with-binary-labels-bias.csv")    
elif dataset == "SUBJ":
    dc = SUBJ(dataset)
elif dataset == "MPQA":
    dc = MPQA(dataset,"subj_obj_sentences.txt")


dc.preprocess()

# BEFORE TRANSLATION
if not args.translated:      
    dc.generate_sentences()

# AFTER TRANSLATION
else:
    dc.ensemble()