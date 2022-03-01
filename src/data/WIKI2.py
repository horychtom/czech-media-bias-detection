from datasets import load_dataset,Dataset,concatenate_datasets

from DataCreator import DataCreator


class WIKI2(DataCreator):
    
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname = "wiki2-cs.csv"
        self.cols = ["id","pretok","posttok","pre","post"]
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
        biased = load_dataset("csv",data_files=self.cs_path + "corpus.biased",
                    delimiter="\t",column_names=self.cols)['train']
        unbiased = load_dataset("csv",data_files=self.cs_path + "corpus.unbiased",
                    delimiter="\t",column_names=self.cols)['train']
        wordbiased = load_dataset("csv",data_files=self.cs_path + "corpus.wordbiased",
                    delimiter="\t",column_names=self.cols)['train']

        self.sentences = biased['pre'] + biased['post']
        self.labels = [1]*len(biased['pre']) + [0]*len(biased['post'])

        self.sentences_w = wordbiased['pre'] + wordbiased['post']
        self.labels_w = [1]*len(wordbiased['pre']) + [0]*len(wordbiased['post'])

        self.sentences_n = unbiased['pre']
        self.labels_n = [0]*len(unbiased['pre'])



    #this is the only original czech data no need 
    #to generate sentences to translate
    def generate_sentences(self):
        self.ensemble()

    def ensemble(self):
        """Outputs three datasets:
            wiki2.csv: 6k of biased unbiased versions of 3k sentences
            wiki2_word.csv: 3.5k subset of wiki2.csv with only changes in one word
            wiki2_neutral.csv: 7.5k unbiased sents for possible sampling
        """
        wiki2 = Dataset.from_dict({"sentence":self.sentences,"label":self.labels})
        wiki2_word = Dataset.from_dict({"sentence":self.sentences_w,"label":self.labels_w})
        wiki2_neutral = Dataset.from_dict({"sentence":self.sentences_n,"label":self.labels_n})

        wiki2 = wiki2.shuffle(seed=self.seed)
        wiki2_word = wiki2_word.shuffle(seed=self.seed)
        wiki2_neutral = wiki2_neutral.shuffle(seed=self.seed)

        wiki2.to_csv(self.cs_path + "wiki2.csv",index=False)
        wiki2_word.to_csv(self.cs_path + "wiki2_word.csv",index=False)
        wiki2_neutral.to_csv(self.cs_path + "wiki2_neutral.csv",index=False)



