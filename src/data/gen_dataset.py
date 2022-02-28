import argparse

from SUBJ import SUBJ
from MPQA import MPQA
from CWhard import CWhard
from BASIL import BASIL
from UAcrisis import UAcrisis

parser = argparse.ArgumentParser(description='Data preprocessing script. Outputs the data in simple format.')
parser.add_argument("-t","--translated", help="Toggle if you already have translated data", action="store_true")
parser.add_argument("-d","--dataset", help="dataset name",required=True)

args = parser.parse_args()

dataset = args.dataset

if dataset == "UA-crisis":
    dc = UAcrisis(dataset)
elif dataset == "SUBJ":
    dc = SUBJ(dataset)
elif dataset == "MPQA":
    dc = MPQA(dataset)
elif dataset == "CW-HARD":
    dc = CWhard(dataset)
elif dataset == "BASIL":
    dc = BASIL(dataset)
else:
    print("Error, no suitable dataset selected.")


dc.preprocess()

# BEFORE TRANSLATION
if not args.translated:      
    dc.generate_sentences()

# AFTER TRANSLATION
else:
    dc.ensemble()