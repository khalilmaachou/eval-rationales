import json
import torch
from utils import create_dataset_huggingface, create_dataset, CustomDataset, create_batch, summarize_attentions, get_token_offsets, show_text_with_normalized_scores_and_features
from methods.default import DefaultMethod
from torch.utils.data import DataLoader, Dataset
import numpy as np



class AttentionExplainer(DefaultMethod):
    
    def __init__(self, model, dataset, threshold, model_type="transformer_foreginer", query="What is the sentiment of this review?"):
        super(AttentionExplainer, self).__init__(model, dataset, threshold, query)
        self.model_type = model_type
    
    def scores(self, dataset, mean=True):
        inputs = dataset['text']
        
        if self.model_type == "transformer_huggingfaces":
            
            inputs_in, always_kp_ma, _ = self.model.create_batchs(inputs,1, self.query)
            offsets = get_token_offsets(inputs, self.model.tokenizer)
            
            features = []
            probas, predicted_labels, attention, always_keep_masks = [], [], [], []
            i=0
            for batch in inputs_in:
                outputs = self.model.forward(document=batch, metadata=None, label=torch.ones(1, dtype=torch.int64).to(self.model.device))
                probas.extend(outputs["probs"].detach().cpu())
                predicted_labels.extend(outputs["predicted_labels"].detach().cpu())
                attn = summarize_attentions(outputs["attentions"].detach().cpu())
                attention.append(attn.data[offsets[i]].tolist())
                always_keep_mask = always_kp_ma[i][:attn.data[offsets[i]].size()[0]]
                always_keep_masks.append(always_keep_mask)
                i+=1

        else:
            features = []
            inp_stream = {}
            inp_stream["model"] = self.model.archive
            inp_stream["predictor"] = self.model.pred

            inputs_in, always_keep_masks, kept_tokens = create_batch(inputs, 1, inp_stream, self.model.device, self.query)
            
            probas, predicted_labels, attention = [], [], []
            for batch in inputs_in:
                outputs = self.model.model._forward(batch, None, label=torch.ones(len(batch['bert']), dtype=torch.int64).to(self.model.device))
                probas.extend(outputs["probs"].detach().cpu())
                predicted_labels.extend(outputs["predicted_labels"].detach().cpu())
                attention.append(outputs["attentions"].detach())
            
            attentions = []
            i=0
            for attn in attention : 
                attn1 = attn[:, :kept_tokens[i].shape[1]] * (1 - kept_tokens[i]).float()
                inter_attn = attn1 / attn1.sum(-1, keepdim=True)
                attentions.append(inter_attn.cpu().data.tolist())
                i+=1

            
            attention = [item for sublist in attentions for item in sublist]
        
        if self.query != None:
            query_words = self.query.split()
            metadatas= []
            for i in range(len(inputs)):
                metadata = {}
                metadata['always_keep_mask'] = np.array(always_keep_masks[i])
                metadata['convert_tokens_to_instance'] = self.model.convert_tokens_to_instance
                metadata['tokens'] = (inputs[i].split() + ["SEP"] + query_words + ["SEP"])[:len(always_keep_masks[i])]
                metadatas.append(metadata)
        else:
            metadatas= []
            for i in range(len(inputs)):
                metadata = {}
                metadata['always_keep_mask'] = np.array(always_keep_masks[i])
                metadata['convert_tokens_to_instance'] = self.model.convert_tokens_to_instance
                metadata['tokens'] = (inputs[i].split() + ["SEP"])[:len(always_keep_masks[i])]
                metadatas.append(metadata)
        
        return {"features":features, "scores":attention, "predicted_labels": predicted_labels, "probas": probas, "always_keep_masks": always_keep_masks, "metadatas":metadatas}
            

    
    def visualize(self, dataset, top_k=None):
        
        score_dict = self.scores(dataset)
        
        for i in range(len(score_dict["scores"])):
            saliency = score_dict["scores"][i]
            saliency = saliency + [0.0] * (len(dataset["text"][i].split()) - len(saliency))
            pos = np.array(saliency) > 0
            vect = np.array(saliency)[pos]
            args = np.argsort(vect)[::-1]
            
            if top_k != None:
                vect[args[top_k:]] = 0.0
                saliency = np.array(saliency)
                saliency[args[top_k:]] = 0.0
            
            features = np.array(dataset["text"][i].split())
            features = features[pos]
            
            show_text_with_normalized_scores_and_features(
                dataset["text"][i],
                np.array(saliency),
                features[args],
                vect[args],
                score_dict["predicted_labels"][i],
                i
            )
