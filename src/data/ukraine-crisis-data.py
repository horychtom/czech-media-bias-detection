from datasets import load_dataset,Dataset

import argparse
import sys
import os

PATH=os.getcwd().split('/src')[0]
sys.path.insert(1, PATH)

DATA_EN_PATH = PATH + "/data/EN/raw/UA-crisis/"
DATA_CS_PATH = PATH + "/data/CS/raw/UA-crisis/"

parser = argparse.ArgumentParser(description='Ukraine transform script.')
parser.add_argument("--translated", help="Toggle if you already have translated data", action="store_true")
args = parser.parse_args()

data = load_dataset('csv',data_files=DATA_EN_PATH + "sentences-with-binary-labels-bias.csv",sep=",",
                    column_names=["sentence","label"])

# BEFORE TRANSLATION
if not args.translated:      
    #texts to translate
    with open(DATA_EN_PATH + "texts.txt","w") as f:
        for sentence in data['train']['sentence']:
            f.write(sentence+"\n")

    with open(DATA_EN_PATH + "labels.txt","w") as f:
        for label in data['train']['label']:
            f.write(str(label)+"\n")

# AFTER TRANSLATION
else:
    with open(DATA_CS_PATH + "texts-cs.txt","r") as f:
        sentences = f.read().splitlines()

    cs_dataset = Dataset.from_dict({"sentence": sentences,"label":data['train']['label']})
    cs_dataset.to_csv(DATA_CS_PATH+"ua-crisis-cs.csv",index=False)