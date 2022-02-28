import argparse

from SUBJ import SUBJ
from DataCreator import DataCreator
from MPQA import MPQA
from CWhard import CWhard

parser = argparse.ArgumentParser(description='Ukraine transform script.')
parser.add_argument("-t","--translated", help="Toggle if you already have translated data", action="store_true")
parser.add_argument("-d","--dataset", help="dataset name",required=True)

args = parser.parse_args()

dataset = args.dataset

if dataset == "UA-crisis":
    dc = DataCreator(dataset)
elif dataset == "SUBJ":
    dc = SUBJ(dataset)
elif dataset == "MPQA":
    dc = MPQA(dataset)
elif dataset == "CW-HARD":
    dc = CWhard(dataset)


dc.preprocess()

# BEFORE TRANSLATION
if not args.translated:      
    dc.generate_sentences()

# AFTER TRANSLATION
else:
    dc.ensemble()