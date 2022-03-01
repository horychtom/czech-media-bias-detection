import os
import sys
from datasets import load_dataset, Dataset


class DataCreator():
    def __init__(self, dataset_name: str):
        self.PATH = os.getcwd().split('/src')[0]
        sys.path.insert(1, self.PATH)
        self.cs_path = self.PATH + "/data/CS/raw/" + dataset_name + "/"
        self.en_path = self.PATH + "/data/EN/raw/" + dataset_name + "/"
        self.cols = ['sentence', 'label']
        self.seed = 42
        self.fname = None
        self.data = None
        self.sentences = None
        self.labels = None

    def preprocess(self):
        """ Creates sentences and labels from english version of dataset
        """
        self.data = load_dataset('csv', data_files=self.en_path + self.fname,
                                 sep=",", column_names=["sentence", "label"])['train']
        self.sentences = self.data['sentence']
        self.labels = self.data['label']

    def generate_sentences(self):
        """ Generates text files with plain sentences ready to
        translate via DeepL
        """
        i=0
        with open(self.en_path + "texts.txt", "w") as f:
            for sentence in self.sentences:
                f.write(sentence+"\n")

    def ensemble(self):
        """ Expects text-cs.txt file to be placed in czech directory
        of particular dataset. Ensembles czech sentences with labels from 
        original dataset
        """
        with open(self.cs_path + "texts-cs.txt", "r") as f:
            sentences_cs = f.read().splitlines()

        cs_dataset = Dataset.from_dict(
            {"sentence": sentences_cs, "label": self.labels})
        cs_dataset.to_csv(self.cs_path+self.fname, index=False)
