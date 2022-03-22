from datasets import load_dataset, Dataset

from DataCreator import DataCreator


class MPQA(DataCreator):

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname = "subj_obj_sentences.txt"

    def preprocess(self):
        self.data = load_dataset('csv', data_files=self.en_path + self.fname,
                                 sep=",", column_names=self.cols)['train']
        self.sentences = [s.strip("b").strip("\"").strip('\'')
                          for s in self.data['sentence']]

        mapping = {"objective": 0, "subjective": 1}
        self.labels = [mapping[key] for key in self.data['label']]

    def ensemble(self):
        """ Expects text-cs.txt file to be placed in czech directory
        of particular dataset. Ensembles czech sentences with labels from 
        original dataset
        """
        with open(self.cs_path + "texts-cs.txt", "r") as f:
            sentences_cs = f.read().splitlines()

        cs_dataset = Dataset.from_dict(
            {"sentence": sentences_cs, "label": self.labels})
        cs_dataset = cs_dataset.filter(lambda x: len(x['sentence']) > 20)
        cs_dataset.to_csv(self.cs_path+self.fname, index=False)