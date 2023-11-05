import json
from utils import create_dataset_huggingface, create_dataset, CustomDataset, features_accuracy
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class DefaultMethod:
    
    def __init__(self, model, dataset, threshold, query="What is the sentiment of this review?"):
        self.model = model
        self.dataset = dataset
        self._threshold = threshold
        self._aopc_thresholds = [0.003, 0.01, 0.05, 0.1, 0.2, 0.5]
        self.query = query
    
    def generate_comprehessiveness_metrics(self, scorer_dict, dataset):

        metrics_scores = {}
        inputs = dataset['text']

        metadatas = scorer_dict["metadatas"]
            
            
        with torch.no_grad():
            torch.cuda.empty_cache()
            
            #Suffieciency
            document = self.model.regenerate_tokens(scorer_dict["scores"], metadatas, self._threshold, None)

            probas = []
            
            for batch in document:
                outputs = self.model.forward(document=batch, metadata=None, label=torch.ones(1, dtype=torch.int64).to(self.model.device))
                probas.extend(outputs["probs"].detach().cpu())
            
            metrics_scores["sufficiency_classification_scores"] = probas 
            
            del outputs

            torch.cuda.empty_cache()

            metrics_scores["sufficiency_aopc_scores"] = {}
            metrics_scores["comprehensiveness_aopc_scores"] = {}
            
            for t in self._aopc_thresholds:
                
                document = self.model.regenerate_tokens(scorer_dict["scores"], metadatas, t, None)
                
                probas = []

                for batch in document:
                    outputs = self.model.forward(document=batch, metadata=None, label=torch.ones(1, dtype=torch.int64).to(self.model.device))
                    probas.extend(outputs["probs"].detach().cpu())
                
                metrics_scores["sufficiency_aopc_scores"][t] = probas
                del outputs
                del probas
            
            torch.cuda.empty_cache()
            

            document = self.model.remove_tokens(scorer_dict["scores"], metadatas, self._threshold, None)

            
            probas = []
            
            for batch in document:
                outputs = self.model.forward(document=batch, metadata=None, label=torch.ones(1, dtype=torch.int64).to(self.model.device))
                probas.extend(outputs["probs"].detach().cpu())
            
            metrics_scores["comprehensiveness_classification_scores"] = probas
            
            del outputs

            torch.cuda.empty_cache()
            
            for t in self._aopc_thresholds:
                
                document = self.model.remove_tokens(scorer_dict["scores"], metadatas, t, None)

                probas = []

                for batch in document:
                    outputs = self.model.forward(document=batch, metadata=None, label=torch.ones(1, dtype=torch.int64).to(self.model.device))
                    probas.extend(outputs["probs"].detach().cpu())
                
                metrics_scores["comprehensiveness_aopc_scores"][t] = probas
                del outputs
                del probas
            
            torch.cuda.empty_cache()
            
            out_dict = {}
            out_dict["comprehensiveness_probs"] = {}
            out_dict["sufficiency_probs"] = {}
            
            for t in self._aopc_thresholds:
                out_dict["comprehensiveness_probs"][t] = []
                out_dict["sufficiency_probs"][t] = []
                
                for i in range(len(metrics_scores["comprehensiveness_classification_scores"])):
                    out_dict["comprehensiveness_probs"][t].append(metrics_scores["comprehensiveness_aopc_scores"][t][i][scorer_dict["predicted_labels"][i]].detach().cpu().tolist())
                    out_dict["sufficiency_probs"][t].append(metrics_scores["sufficiency_aopc_scores"][t][i][scorer_dict["predicted_labels"][i]].detach().cpu().tolist())
                    
                    
            out_dict["original_probs"] = []
            for i in range(len(metrics_scores["comprehensiveness_classification_scores"])):
                    out_dict["original_probs"].append(scorer_dict["probas"][i][scorer_dict["predicted_labels"][i]].detach().cpu().tolist())
            
            return out_dict
            
            

        
        
    
    def scores(self, inputs, labels=None):
        raise NotImplementedError
        
    def generate_token_metrics(self, scorer_dict, dataset):
        metrics_scores = {}
        score = scorer_dict["scores"]
        metadatas = scorer_dict["metadatas"]
                
        features, gold_list= [], []
        for i in range(len(dataset['evidences'])) :

            args = np.argsort(score[i])[::-1]

            features.append(np.array(metadatas[i]['tokens'])[args.astype(int)])
            gold_list.append(self.evidence_preprocessing(dataset['evidences'][i]))
        
        for p in range(10,110,10):
            accuracy = 0
            f1_mesure = 0
            recall = 0
            for i in range(len(features)):
                metric = features_accuracy(gold_list[i], features[i], k_features=p)
                accuracy = accuracy + metric[0]
                f1_mesure = f1_mesure + metric[2]
                recall = recall + metric[1]
            
            print(str(p)+"%")
            print("Global accuracy : " + str(accuracy/len(features)))
            print("Global f1_mesure : " + str(f1_mesure/len(features)))
            print("Global recall : " + str(recall/len(features)))
                
    def evidence_preprocessing(self, evidences):
        tokenized_liste = []
        for e in evidences:
            tokenized_liste.extend(e.split())

        return tokenized_liste
   
    
