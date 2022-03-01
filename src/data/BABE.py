from datasets import load_dataset, Dataset

from DataCreator import DataCreator


class BABE(DataCreator):

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname = "SG2.csv"

    def preprocess(self):
        self.data = load_dataset(
            'csv', data_files=self.en_path + self.fname, sep=";")['train']
        
        self.data = self.data.filter(lambda row: row['label_bias'] != 'No agreement')
        mapping = {'Non-biased':0, 'Biased':1}
        self.sentences = self.data['text']
        self.labels = [mapping[key] for key in self.data['label_bias']]
