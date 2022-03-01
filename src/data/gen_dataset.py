import argparse

from SUBJ import SUBJ
from MPQA import MPQA
from CWhard import CWhard
from BASIL import BASIL
from UAcrisis import UAcrisis
from WIKI1 import WIKI1
from WIKI2 import WIKI2
from BABE import BABE
from WNC import WNC

parser = argparse.ArgumentParser(
    description='Data preprocessing script. Outputs the data in simple format.')
parser.add_argument("-t", "--translated",
                    help="Toggle if you already have translated data", action="store_true")
parser.add_argument("-d", "--dataset", help="dataset name", required=True)

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
elif dataset == "WIKI1":
    dc = WIKI1(dataset)
elif dataset == "WIKI2":
    dc = WIKI2(dataset)
elif dataset == "BABE":
    dc = BABE(dataset)
elif dataset == "WNC":
    dc = WNC(dataset)
else:
    print("Error, no suitable dataset selected.")

dc.preprocess()
if not args.translated:
    dc.generate_sentences()
else:
    dc.ensemble()
