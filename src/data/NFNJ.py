from datasets import load_dataset, Dataset
from DataCreator import DataCreator

import pandas as pd
import numpy as np


class NFNJ(DataCreator):

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname = "nfnj.csv"
        self.obj = None
        self.subj = None
        #mapping {1:'neutral',2:'slightly biased but acceptable',3:'biased',4:'very biased'}

    def preprocess(self):
        data = load_dataset(
            "csv", data_files=self.en_path + "data.csv")['train']
        data = data.remove_columns(['id_event', 'event', 'date_event', 'source', 'source_bias',
                                   'url', 'ref', 'reftitle', 'ref_url', 'article_bias', 'preknow', 'reftext', 'docbody'])
        data = data.to_pandas()
        #data has format where each column is a sentence and has 20 sentenes + title
        sent_indices = ['doctitle'] + list(map(lambda x: "s"+str(x),[n for n in range(20)]))
        label_indices = ['t'] + list(map(lambda x: str(x),[n for n in range(20)]))

        #extract same sentences. Each set of sentences is in data multiple times
        #because of the multiple annotations
        s = data['s0'][0]
        sentences = list(data[sent_indices].iloc[0])
        group = 1

        for i in range(len(data)):
            snew = data['s0'][i]
            if snew == s:
                group+=1
            else:
                sentences = sentences + list(data[sent_indices].iloc[i])
                group=1
            s = snew

        # same for labels, aggregate labels via mean
        labels = data.groupby(['id_article'])[label_indices].mean()
        labels = labels.to_numpy().flatten()

        #labels normalized to unbiased/biased
        labels = list(map(lambda x: 0 if x <= 2 else 1,labels))

        #strip number from body sentences
        sentences_processed = []
        for sent in sentences:
            if sent is None:
                sentences_processed.append("")
            elif sent[0] == '[':
                sentences_processed.append(sent[5:])
            else:
                sentences_processed.append(sent)

        self.sentences = sentences_processed
        self.labels = labels
