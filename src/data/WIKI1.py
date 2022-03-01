from datasets import load_dataset,Dataset,concatenate_datasets

from DataCreator import DataCreator


class WIKI1(DataCreator):
    
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.fname = "wiki1-cs.csv"
        self.cols = ['label','sentence']

    def preprocess(self):
        pass

    #this is the only original czech data no need 
    #to generate sentences to translate
    def generate_sentences(self):
        self.ensemble()

    def ensemble(self):
        label_to_num = lambda data : {'label':0 if data['label'] == '__label__neutral' else 1,'sentence':data['sentence']}
        cols = ['label','sentence']

        train = load_dataset('csv',data_files=self.cs_path + "CS-train.txt",sep='\t',
                             column_names=self.cols)['train']
        dev = load_dataset('csv',data_files=self.cs_path + "CS-dev.txt",sep='\t',
                             column_names=self.cols)['train']
        test = load_dataset('csv',data_files=self.cs_path + "CS-test.txt",sep='\t',
                             column_names=self.cols)['train']

        full = concatenate_datasets([train,dev,test])
        full = full.map(label_to_num)
        full = full.shuffle(seed=self.seed)
        full = Dataset.from_dict({"sentence":full['sentence'],"label":full['label']})
        full.to_csv(self.cs_path+self.fname,index=False)
            

