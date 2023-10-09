from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import numpy as np
from model.default import DefaultModel
import spacy
import torch




from utils import create_batches

class Linear_Model(DefaultModel):
    
    def __init__(self, dataset):
        super(Linear_Model, self).__init__(None, dataset)
        self.tokenizer = TfidfVectorizer(stop_words=None, min_df=0.0,max_df=1.0 , token_pattern = r'[a-zA-Z]+')
        self.model = self.train()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.convert_tokens_to_instance = None
        
    def train(self, a_venir=None):
        #X_train = self.dataset['train']['text']
        X_test = self.dataset['text']
        
        #y_train = self.dataset['train']['label']
        y_test = self.dataset['label']
        
        X_train_bow = self.tokenizer.fit_transform(X_test) # fit train
        X_test_bow = self.tokenizer.transform(X_test) # transform test
        
        return LogisticRegression().fit(X_train_bow, y_test)
    
    def predictor(self, inputs):        
        return self.model.predict_proba(self.tokenizer.transform(inputs))
    
    def process_text(self, text):
        
        nlp = spacy.load("en_core_web_sm")

        # tokenize text
        text_tokens = nlp(text)    

        #Import the english stop words list from NLTK
        stopwords_english = stopwords.words('english') 

        #Creating a list of stems of words in tweet
        text_stem = []
        for word in text_tokens:
            if word.pos_ != "PUNCT" and word.text not in stopwords_english:
                text_stem.append(word.lemma_.lower().strip())

        return text_stem
    
    def data_preparation(self, dataset):
        preprocessed_data = []
        for data in dataset:
            preprocessed_data.append(' '.join(self.process_text(data["text"])))
        
        return preprocessed_data
    
    def lime_tokenizer(self, metadata, always_keep_mask, filtered_tokens):
        new_tokens = filtered_tokens
        new_tokens = [t for i, t in enumerate(metadata['tokens']) if i in new_tokens or always_keep_mask[i] == 1]
        document = self.tokenizer.transform(
                [" ".join(new_tokens)]
        )
        return document
    
    def lime_tokenizer_viz(self, metadata, always_keep_mask, filtered_tokens):
        new_tokens = filtered_tokens
        new_tokens = [t for i, t in enumerate(metadata['tokens']) if t in new_tokens or always_keep_mask[i] == 1]
        document = self.tokenizer.transform(
                [" ".join(new_tokens)]
        )
        return document
    
    def create_dataset(self, inputs):
        result = []
        for inp in inputs:
            result.append(self.tokenizer.transform([inp]))

        return result
    
    def remove_tokens(self, attentions, metadata, threshold, labels):
        attentions_cpu = np.array(attentions)
        sentences = [x["tokens"] for x in metadata]
        instances = []
        for b in range(attentions_cpu.shape[0]):
            sentence = [x for x in sentences[b]]
            always_keep_mask = metadata[b]['always_keep_mask']
            attn = attentions_cpu[b][: len(sentence)] + always_keep_mask * -10000
            max_length = math.ceil((1 - always_keep_mask).sum() * threshold)
            top_ind = np.argsort(attn)[:-max_length]
            new_tokens = ' '.join([x for i, x in enumerate(sentence) if i in top_ind or always_keep_mask[i] == 1])
            instances.append(self.tokenizer.transform([new_tokens]))
        
        return instances
    
    def regenerate_tokens(self, attentions, metadata, threshold, labels):
        attentions_cpu = np.array(attentions)
        sentences = [x["tokens"] for x in metadata]
        instances = []
        for b in range(len(attentions_cpu)):
            sentence = [x for x in sentences[b]]
            always_keep_mask = metadata[b]['always_keep_mask']
            attn = attentions_cpu[b][: len(sentence)] + always_keep_mask * -10000
            max_length = math.ceil((1 - always_keep_mask).sum() * threshold)
            top_ind = np.argsort(attn)[-max_length:]
            new_tokens = ' '.join([x for i, x in enumerate(sentence) if i in top_ind or always_keep_mask[i] == 1])
            instances.append(self.tokenizer.transform([new_tokens]))
            
        return instances
    
    def forward(self, document, metadata=None, label=None):

        probas = self.model.predict_proba(document)
        probas = torch.tensor(probas).to(self.device)
        output={}

        output["probs"]=probas
        output["predicted_labels"]=probas.argmax(-1)
        output["attentions"] = None
        
        return output
    
    def create_batchs(self, inputs, batch_size, query):
        always_keep_masks = []
        kept_tokens = []
        for docwords in inputs:
            
            always_keep_mask = [0] * len(docwords.split(" "))
            always_keep_mask += [1]
            
            if query != None:
                always_keep_mask += [1] * (len(query.split(" ")) + 1)
                
            always_keep_masks.append(always_keep_mask)
            kept_tokens.append(torch.tensor([always_keep_mask]).to(self.device).detach())

        return self.tokenizer.transform(inputs), always_keep_masks, kept_tokens
