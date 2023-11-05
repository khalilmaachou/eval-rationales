import torch
from utils import create_batch, summarize_attentions, get_token_offsets, show_text_with_normalized_scores_and_features
from methods.default import DefaultMethod
import time
import numpy as np



class RandomExplainer(DefaultMethod):
    
    def __init__(self, model, dataset, threshold, model_type="transformer_foreginer", query="What is the sentiment of this review?"):
        super(RandomExplainer, self).__init__(model, dataset, threshold, query)
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
                seed = time.time()
                torch.manual_seed(seed)
                attn = torch.rand_like(attn).detach()
                attention.append(attn[offsets[i]].tolist())
                temp = [1] + always_kp_ma[i]
                always_keep_mask = temp[:attn.data[offsets[i]].size()[0]]
                always_keep_masks.append(always_keep_mask)
                i+=1

            metadatas= []
            if self.query != None:
                query_words = ["[SEP]"] + self.query.split() + ["[SEP]"]
            else:
                query_words = ["[SEP]"]
                
            for i in range(len(inputs)):
                metadata = {}
                metadata['always_keep_mask'] = np.array(always_keep_masks[i])
                metadata['convert_tokens_to_instance'] = self.model.convert_tokens_to_instance
                metadata['tokens'] = (list(filter(lambda x: bool(len(x)), inputs[i].strip().split(' '))) + query_words)[:len(always_keep_masks[i])]
                metadata['all_tokens'] = list(filter(lambda x: bool(len(x)), inputs[i].strip().split(' '))) + query_words
                metadatas.append(metadata)


        else:
            features = []

            inp_stream = {}
            inp_stream["model"] = self.model.archive
            inp_stream["predictor"] = self.model.pred

            inputs_in, always_keep_masks, kept_tokens = create_batch(inputs, 1, inp_stream, self.model.device, self.query)
            
            probas, predicted_labels, attention = [], [], []
            for batch in inputs_in:
                outputs = self.model.model._forward(batch, None, label=torch.ones(len(batch['bert']), dtype=torch.int64).to('cuda:0'))
                probas.extend(outputs["probs"].detach().cpu())
                predicted_labels.extend(outputs["predicted_labels"].detach().cpu())
                attention.append(outputs["attentions"].detach())
            
            attentions = []
            i=0
            for attn in attention : 
                attn1 = attn[:, :kept_tokens[i].shape[1]] * (1 - kept_tokens[i]).float()
                seed = time.time()
                torch.manual_seed(seed)
                attentions.append(torch.rand_like(attn1).cpu().data.tolist())
                i+=1
            
            attention = [item for sublist in attentions for item in sublist]
        
            metadatas= []
            if self.query != None:
                query_words = ["[SEP]"] + self.query.split() + ["[SEP]"]
            else:
                query_words = ["[SEP]"]
                
            for i in range(len(inputs)):
                metadata = {}
                metadata['always_keep_mask'] = np.array(always_keep_masks[i])
                metadata['convert_tokens_to_instance'] = self.model.convert_tokens_to_instance
                metadata['tokens'] = (list(filter(lambda x: bool(len(x)), inputs[i].strip().split(' '))) + query_words)
                metadata['all_tokens'] = list(filter(lambda x: bool(len(x)), inputs[i].strip().split(' '))) + query_words
                metadatas.append(metadata)

        return {"features":features, "scores":attention, "predicted_labels": predicted_labels, "probas": probas, "always_keep_masks": always_keep_masks, "metadatas":metadatas}

    def visualize(self, dataset, top_k=None):
        
        score_dict = self.scores(dataset)
        
        for i in range(len(score_dict["scores"])):
            saliency = score_dict["scores"][i] + score_dict["metadatas"][i]["always_keep_mask"] * -10000
            saliency = saliency.tolist() + [0.0] * (len(score_dict["metadatas"][i]["all_tokens"]) - len(saliency))
            saliency = [0.0 if e < 0.0 else e for e in saliency ]

            pos = np.array(saliency) > 0
            vect = np.array(saliency)[pos]
            args = np.argsort(vect)[::-1]
            
            if top_k != None:
                vect[args[top_k:]] = 0.0
                saliency = np.array(saliency)
                saliency[args[top_k:]] = 0.0
            
            #features = np.array(dataset["text"][i].split())
            features = np.array(score_dict["metadatas"][i]["all_tokens"])
            features = features[pos]
            
            show_text_with_normalized_scores_and_features(
                score_dict["metadatas"][i]["all_tokens"],
                np.array(saliency),
                features[args],
                vect[args],
                score_dict["predicted_labels"][i],
                i
            )

