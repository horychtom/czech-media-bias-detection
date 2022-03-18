from datasets import load_dataset, Dataset, concatenate_datasets

from DataCreator import DataCreator


class CWNC(DataCreator):

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname = "cwnc-cs.csv"
        self.cols = ["id", "pretok", "posttok", "pre", "post"]
        self.sentences_w = None
        self.labels_w = None
        self.sentences_n = None
        self.labels_n = None

    def preprocess(self):
        """corpus.biased: pre and post editorial sentences
           corpus.unbiased: pre and post unchanged sentences
           corpus.wordbiased: subset of corpus.biased where 
           only one word was changed
        """
        biased = load_dataset("csv", data_files=self.cs_path + "corpus.biased",
                              delimiter="\t", column_names=self.cols)['train']
        unbiased = load_dataset("csv", data_files=self.cs_path + "corpus.unbiased",
                                delimiter="\t", column_names=self.cols)['train']
        wordbiased = load_dataset("csv", data_files=self.cs_path + "corpus.wordbiased",
                                  delimiter="\t", column_names=self.cols)['train']

        self.sentences = biased['pre'] + biased['post']
        self.labels = [1]*len(biased['pre']) + [0]*len(biased['post'])

        self.sentences_w = wordbiased['pre'] + wordbiased['post']
        self.labels_w = [1]*len(wordbiased['pre']) + \
            [0]*len(wordbiased['post'])

        self.sentences_n = unbiased['pre']
        self.labels_n = [0]*len(unbiased['pre'])

    # this is the only original czech data no need
    # to generate sentences to translate
    def generate_sentences(self):
        self.ensemble()

    def ensemble(self):
        """Outputs three datasets:
            cwnc.csv: 6k of biased unbiased versions of 3k sentences
            cwnc_word.csv: 3.5k subset of cwnc.csv with only changes in one word
            cwnc_neutral.csv: 7.5k unbiased sents for possible sampling
        """
        cwnc = Dataset.from_dict(
            {"sentence": self.sentences, "label": self.labels})
        cwnc_word = Dataset.from_dict(
            {"sentence": self.sentences_w, "label": self.labels_w})
        cwnc_neutral = Dataset.from_dict(
            {"sentence": self.sentences_n, "label": self.labels_n})

        cwnc = cwnc.shuffle(seed=self.seed)
        cwnc_word = cwnc_word.shuffle(seed=self.seed)
        cwnc_neutral = cwnc_neutral.shuffle(seed=self.seed)

        cwnc = cwnc.filter(lambda x: len(x['sentence']) > 30)
        cwnc = cwnc.filter(lambda x: "NPOV" not in x['sentence'])
        cwnc = cwnc.filter(lambda x: "POV" not in x['sentence'])

        cwnc.to_csv(self.cs_path + "cwnc.csv", index=False)
        cwnc_word.to_csv(self.cs_path + "cwnc_word.csv", index=False)
        cwnc_neutral.to_csv(self.cs_path + "cwnc_neutral.csv", index=False)
