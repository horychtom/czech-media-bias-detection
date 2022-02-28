from datasets import load_dataset,Dataset

from DataCreator import DataCreator


class BASIL(DataCreator):

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname = "basil.csv"

    def preprocess(self):
        self.data = load_dataset('csv',data_files=self.en_path + self.fname,sep=",")['train']
        self.data = self.data.filter(lambda row: row['sentence'] != None)
        self.sentences = self.data['sentence']
        self.labels = self.data['lex_bias']

