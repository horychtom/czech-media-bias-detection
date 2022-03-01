from datasets import load_dataset, Dataset

from DataCreator import DataCreator


class SUBJ(DataCreator):

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname = "subj.csv"
        self.obj = None
        self.subj = None

    def preprocess(self):
        with open(self.en_path + "obj_EN.txt", "r") as f:
            self.obj = f.read().splitlines()
        with open(self.en_path + "subj_EN.txt", "r", encoding="utf-8", errors='ignore') as f:
            self.subj = f.read().splitlines()

        self.sentences = self.obj + self.subj
        self.labels = [0]*len(self.obj) + [1]*len(self.subj)

    def ensemble(self):
        with open(self.cs_path + "texts-cs.txt", "r") as f:
            sentences_cs = f.read().splitlines()

        cs_dataset = Dataset.from_dict(
            {"sentence": sentences_cs, "label": self.labels})
        cs_dataset = cs_dataset.shuffle(seed=self.seed)
        cs_dataset.to_csv(self.cs_path+self.fname, index=False)
