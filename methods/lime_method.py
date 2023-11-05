import torch
from lime.lime_text import LimeTextExplainer

from methods.default import DefaultMethod
import math

import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class LimeExplainer(DefaultMethod):
    
    def __init__(self, model, dataset, threshold, model_type="transformer_foreginer", query="What is the sentiment of this review?", class_names=['negative', 'positive']):
        super(LimeExplainer, self).__init__(model, dataset, threshold, query)
        self._J=0
        self.expl = LimeTextExplainer(class_names=class_names,split_expression=" ", bow=False, random_state=42)
        self.model_type = model_type
    
    def scores(self, dataset):
        
        features, attentions = [], []
        labels = torch.ones(1, dtype=torch.int64).to(self.model.device)
        inputs = dataset['text']
        
        inputs_in , always_keep_masks, kept_tokens = self.model.create_batchs(inputs, 1, self.query)           
        
        metadatas = []
        if self.query != None:
            query_tokens = ["[SEP]"] + self.query.split() + ["[SEP]"]
        else:
            query_tokens = ["[SEP]"]
        for i in range(len(inputs)):
            metadata = {}
            always_keep_mask = np.array(always_keep_masks[i])
            metadata['always_keep_mask'] = always_keep_mask
            metadata['convert_tokens_to_instance'] = self.model.convert_tokens_to_instance
            metadata['tokens'] = list(filter(lambda x: bool(len(x)), inputs[i].strip().split(' '))) + query_tokens
            metadatas.append(metadata)
            
        attentions = []
        predicted_labels = []
        probabilite = []
        for batch in inputs_in:
            outputs = self.model.forward(document=batch, metadata=None, label=labels)
            predicted_labels.extend(outputs["predicted_labels"].detach().cpu())
            probabilite.extend(outputs["probs"].detach().cpu())
            del outputs
            torch.cuda.empty_cache()
            
        
        def predic_prob(inp_texts):
                        
            filtered_tokens = [
                    [int(t) for t in selected_tokens.split(" ") if t != "UNKWORDZ"] for selected_tokens in inp_texts
                ]

            #inputs_in, _, _ = create_batch(inp_texts, 1, inp_stream)
            probas = []
            for i in range(len(filtered_tokens)):

                document = self.model.lime_tokenizer(metadatas[self._J], always_keep_masks[self._J], filtered_tokens[i])
                outputs = self.model.forward(document=document, metadata=None, label=torch.ones(1, dtype=torch.int64).to(self.model.device))
                probas.append(outputs["probs"].detach().cpu())

                del outputs
                torch.cuda.empty_cache()



            probas = torch.cat(probas, dim=0)
            return probas
        
        indice=0
        self._J=0
        for row in inputs:
            logging.info("INPUT "+str(indice))
            predicted_label = predicted_labels[indice].cpu().tolist()
                    
            always_keep_mask = always_keep_masks[indice]

            selection_tokens = [i for i, x in enumerate(always_keep_mask) if x != 1]

            num_features = math.ceil(self._threshold * len(selection_tokens))

            
            explanation = self.expl.explain_instance(
                " ".join([str(i) for i in selection_tokens]),
                predic_prob,
                num_features=min(len(selection_tokens), num_features*2),
                labels=(predicted_label,),
                num_samples=1000,
            )
            
            weights=explanation.as_list(predicted_label)
            
            saliency = [0.0 for _ in range(len(metadatas[self._J]['tokens']))]
            for f, w in weights:
                saliency[int(f)] = max(0.0, w)

            
            #Calculate the attention 
            saliency = torch.Tensor([saliency]).to(self.model.device)
            attention = saliency
            k_p = torch.zeros(kept_tokens[indice].shape[1] - attention.shape[1]).to(self.model.device)
            attention = torch.cat((attention[0], k_p)).unsqueeze(0).to(self.model.device)
            
            
            attn1 = attention * (1 - kept_tokens[indice].to(self.model.device)).float()
            inter_attn = attn1 / attn1.sum(-1, keepdim=True)
            attentions.append(inter_attn.cpu().data.tolist())
            indice+=1
            self._J+=1
            
        score = [item for sublist in attentions for item in sublist]
        
        return {"features":features, "scores":score, "predicted_labels": predicted_labels, "probas": probabilite, "always_keep_masks": always_keep_masks, "metadatas":metadatas}
            
    def visualize(self, dataset, top_k=None):
        inputs = dataset["text"]
        labels = torch.tensor(dataset["label"]).to(self.model.device)
        
        inputs_in , always_keep_masks, kept_tokens = self.model.create_batchs(inputs, 1, self.query)
        
        metadatas = []
        if self.query != None:
            query_tokens = ["SEP"] + self.query.split() + ["SEP"]
        else:
            query_tokens = []
        for i in range(len(inputs)):
            metadata = {}
            always_keep_mask = np.array(always_keep_masks[i])
            metadata['always_keep_mask'] = always_keep_mask
            metadata['convert_tokens_to_instance'] = self.model.convert_tokens_to_instance
            metadata['tokens'] = list(filter(lambda x: bool(len(x)), inputs[i].strip().split(' '))) + query_tokens
            metadatas.append(metadata)
            
        predicted_labels = []
        i = 0
        for batch in inputs_in:
            outputs = self.model.forward(document=batch, metadata=None, label=labels[i])
            predicted_labels.extend(outputs["predicted_labels"].detach().cpu())
            del outputs
            torch.cuda.empty_cache()
            i+=1
            
        
        def predic_prob(inp_texts):
            filtered_tokens = [
                    [t for t in selected_tokens.split(" ") if t != "UNKWORDZ"] for selected_tokens in inp_texts
                ]

            #inputs_in, _, _ = create_batch(inp_texts, 1, inp_stream)
            probas = []
            for i in range(len(filtered_tokens)):

                document = self.model.lime_tokenizer_viz(metadatas[self._J], always_keep_masks[self._J], filtered_tokens[i])
                outputs = self.model.forward(document=document, metadata=None, label=torch.ones(1, dtype=torch.int64).to(self.model.device))
                probas.extend(outputs["probs"].detach().cpu().tolist())

                del outputs
                torch.cuda.empty_cache()

            return np.array(probas)
        
        self._J=0
        for inp in inputs:
            predicted_label = predicted_labels[self._J].cpu().tolist()
                    
            always_keep_mask = always_keep_masks[self._J]
        
            selection_tokens = [i for i, x in enumerate(always_keep_mask) if x != 1]

            num_features = math.ceil(self._threshold * len(selection_tokens))

            explanation = self.expl.explain_instance(
                    inp,
                    predic_prob,
                    num_features=min(len(selection_tokens), num_features*2),
                    labels=(predicted_label,),
                    num_samples=1000,
                )
            
            explanation.show_in_notebook()
            explanation.save_to_file("output_"+str(self._J)+".html", labels=(predicted_label,))
            
            self._J+=1
            
        
    
    
    

    
