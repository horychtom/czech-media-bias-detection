from datasets import load_dataset, Dataset, concatenate_datasets

from DataCreator import DataCreator


class WikiBias(DataCreator):

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname = "wikibias.csv"
        self.obj = None
        self.subj = None

    def preprocess(self):
        train = load_dataset("csv", data_files=self.en_path + "train.tsv", delimiter="\t", column_names=[
                             'sentence', 'weird', 'label'])['train']
        dev = load_dataset("csv", data_files=self.en_path + "dev.tsv", delimiter="\t", column_names=[
                           'sentence', 'weird', 'label'])['train']
        test = load_dataset("csv", data_files=self.en_path + "test.tsv", delimiter="\t", column_names=[
                            'sentence', 'weird', 'label'])['train']

        full = concatenate_datasets([train, dev, test])
        full = full.remove_columns(['weird'])
        full.to_csv(self.en_path + self.fname, index=False)
        self.sentences = full['sentence']
        self.labels = full['label']
