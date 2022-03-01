from datasets import load_dataset, Dataset

from DataCreator import DataCreator


class WNC(DataCreator):

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname = "biased_neutral_180k.csv"
        self.cols = ["id", "pretok", "posttok", "pre", "post","tag1","tag2"]
        
    def preprocess(self):
        """corpus.biased: pre and post editorial sentences
           corpus.unbiased: pre and post unchanged sentences
           corpus.wordbiased: subset of corpus.biased where 
           only one word was changed
        """
        biased = load_dataset("csv", data_files=self.en_path + self.fname,
                              delimiter="\t", column_names=self.cols)['train']

        self.sentences = biased['pre'] + biased['post']
        self.labels = [1]*len(biased['pre']) + [0]*len(biased['post'])


    # ALERT! for now changed to en, so i can make some experiments with it
    def ensemble(self):
        dataset = Dataset.from_dict(
            {"sentence": self.sentences, "label": self.labels})
        dataset.to_csv(self.PATH + "/data/EN/processed/WNC/" + "wnc.csv", index=False)