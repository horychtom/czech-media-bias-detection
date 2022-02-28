from datasets import load_dataset,Dataset

from DataCreator import DataCreator


class CWhard(DataCreator):
    
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname =  "cw-hard.csv"
        self.biased = None
        self.neutral = None

    def preprocess(self):
        with open(self.en_path + "statements_biased","r") as f:
            self.biased = f.read().splitlines()
        with open(self.en_path + "statements_neutral_cw-hard","r") as f:
            self.neutral = f.read().splitlines()

        self.sentences = self.biased + self.neutral
        self.labels = [1]*len(self.biased) + [0]*len(self.neutral)
            
    def ensemble(self):
        with open(self.cs_path + "texts-cs.txt","r") as f:
            sentences_cs = f.read().splitlines()

        cs_dataset = Dataset.from_dict({"sentence": sentences_cs,"label":self.labels})
        cs_dataset = cs_dataset.shuffle()
        cs_dataset.to_csv(self.cs_path+self.fname,index=False)

