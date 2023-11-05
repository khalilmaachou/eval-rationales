import torch
from model.default import DefaultModel
import torch.nn.functional as F

from model.default import DefaultModel
import numpy as np
import math
from utils import create_batch_huggingface
import torch





class HuggingFace_Transformer(DefaultModel):
    
    def __init__(self, dataset, embedding_layer, model, tokenizer):
        
        super(HuggingFace_Transformer, self).__init__(None, dataset)
        self.model = model
        self.tokenizer = tokenizer
        self._embedding_layer = embedding_layer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.convert_tokens_to_instance = None
        self.model.to(self.device)
        
    def train(self, a_venir=None):
        return None
    
    def forward(self, document, metadata=None, label=None):        
        outputs = self.model(**document)
        probas = F.softmax(outputs.logits, dim=1).detach()
        output = {}

        output["probs"]=probas
        output["predicted_labels"]=probas.argmax(-1)
        output["attentions"] = outputs["attentions"][len(outputs["attentions"])-1]
        
        del outputs
        return output
    
    def predictor(self, input_ids, attention_mask):
        outputs = self.model(input_ids,attention_mask=attention_mask)
        probas = F.softmax(outputs.logits, dim=1)
            
        if(len(probas[0])==3):
            probas = np.array([[p[0], p[1]+p[2]] for p in probas])

        return probas
    
    def lime_tokenizer(self, metadata, always_keep_mask, filtered_tokens):
        new_tokens = [t for i, t in enumerate(metadata['tokens']) if i in filtered_tokens or always_keep_mask[i] == 1]
        document = self.tokenizer(
                text=" ".join(new_tokens), return_tensors="pt", truncation=True, max_length=512
        )
        return {k: v.to(self.device) for k, v in document.items()}
    
    def lime_tokenizer_viz(self, metadata, always_keep_mask, filtered_tokens):
        new_tokens = [t for i, t in enumerate(metadata['tokens']) if t in filtered_tokens or always_keep_mask[i] == 1]
        document = self.tokenizer(
                text=" ".join(new_tokens), return_tensors="pt", truncation=True, max_length=512
        )
        return {k: v.to(self.device) for k, v in document.items()}
    
    def create_dataset(self, inputs):
        result = []
        for inp in inputs:
            result.append({k: v.to(self.device).detach() for k, v in self.tokenizer(inp,return_tensors="pt", truncation=True, max_length=512).items()})

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
            instances.append({k: v.to(self.device) for k, v in self.tokenizer(new_tokens,return_tensors="pt", truncation=True, max_length=512).items()})
        
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
            instances.append({k: v.to(self.device) for k, v in self.tokenizer(new_tokens,return_tensors="pt", truncation=True, max_length=512).items()})
            
        return instances
    
    def create_batchs(self, inputs, batch_size, query):
        kwarg = {}
        kwarg['return_tensors']="pt"
        kwarg['truncation']=True
        kwarg['max_length']=512
        return create_batch_huggingface(inputs, query, self.tokenizer, self.device, kwarg)