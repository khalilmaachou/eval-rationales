from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import LabelField, MetadataField, TextField, SequenceLabelField
from allennlp.data import Instance
from allennlp.data.dataset import Batch

from utils import create_batches, CustomDataset, create_dataset
from model.default import DefaultModel
import numpy as np
import math
from utils import create_dataset_huggingface, create_dataset, CustomDataset, create_batch
import torch


class Foreginer_Transformer(DefaultModel):
    
    def __init__(self, archive_url, predictor_type, dataset):
        super(Foreginer_Transformer, self).__init__(None, dataset)
        self.archive = load_archive(archive_url, cuda_device=0)
        self.model = self.archive.model
        self.pred = Predictor.from_archive(self.archive, predictor_type)
        self.token_indexer = self.pred._dataset_reader._token_indexers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train(self, a_venir=None):
        return None
    
    def predictor(self, inputs):        
        self.model.to(self.device)
        inp_stream = {}
        inp_stream["model"] = self.model
        inp_stream["predictor"] = self.pred
        dataset, _ = create_dataset(inputs, inp_stream, self.device)
        dataset = CustomDataset(dataset)

        # Créer le DataLoader pour grouper les exemples en batches
        batch_size = 4
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        probas = []
        
        for batch in dataloader:
            outputs = self.model._forward(batch, None, label=torch.ones(len(batch["bert"]), dtype=torch.int64).to(self.device))
            probas.append(outputs["probs"].detach().cpu())

            del outputs
            del batch
            torch.cuda.empty_cache()

        probas = torch.cat(probas, dim=0)
        return probas

    def remove_tokens(self, attentions, metadata, threshold, labels):
        attentions_cpu = np.array(attentions)
        sentences = [x["tokens"] for x in metadata]
        batchs = []
        
        for b in range(attentions_cpu.shape[0]):
            instances = []
            sentence = [x for x in sentences[b]]
            always_keep_mask = metadata[b]['always_keep_mask']
            attn = attentions_cpu[b][: len(sentence)] + always_keep_mask * -10000
            max_length = math.ceil((1 - always_keep_mask).sum() * threshold)

            top_ind = np.argsort(attn)[:-max_length]
            new_tokens = [Token(x) for i, x in enumerate(sentence) if i in top_ind or always_keep_mask[i] == 1]
            instances += metadata[0]["convert_tokens_to_instance"](new_tokens, None)

            batch = Batch(instances)
            batch.index_instances(self.archive.model._vocabulary)
            padding_lengths = batch.get_padding_lengths()

            batch = batch.as_tensor_dict(padding_lengths)
            batchs.append({k: v.to(self.device) for k, v in batch["document"].items()})
            
            del batch
            del instances
            
        return batchs
    
    def regenerate_tokens(self, attentions, metadata, threshold, labels):
        attentions_cpu = np.array(attentions)
        sentences = [x["tokens"] for x in metadata]
        instances = []
        batchs = []
        for b in range(len(attentions_cpu)):
            instances = []
            sentence = [x for x in sentences[b]]
            always_keep_mask = metadata[b]['always_keep_mask']
            attn = attentions_cpu[b][: len(sentence)] + always_keep_mask * -10000
            max_length = math.ceil((1 - always_keep_mask).sum() * threshold)
            top_ind = np.argsort(attn)[-max_length:]
            new_tokens = [Token(x) for i, x in enumerate(sentence) if i in top_ind or always_keep_mask[i] == 1]
            instances += metadata[0]["convert_tokens_to_instance"](new_tokens, None)

            batch = Batch(instances)
            batch.index_instances(self.archive.model._vocabulary)
            padding_lengths = batch.get_padding_lengths()

            batch = batch.as_tensor_dict(padding_lengths)
            batchs.append({k: v.to(self.device) for k, v in batch["document"].items()})
            
            del batch
            del instances
            
        return batchs
    
    
    def convert_tokens_to_instance(self, tokens, labels=None):
        return [Instance({"document": TextField(tokens, self.pred._dataset_reader._token_indexers)})]
    
    def generate_tokens(self, new_tokens, metadata, labels):
        return self.model.generate_tokens(new_tokens=new_tokens, metadata=metadata, labels=labels)
    
    def forward(self, document, metadata, label):
        return self.model._forward(document, None, label=label)
    
    def lime_tokenizer(self, metadata, always_keep_mask, filtered_tokens):
        new_tokens = filtered_tokens
        new_tokens = [Token(t) for i, t in enumerate(metadata['tokens']) if i in new_tokens or always_keep_mask[i] == 1]
        new_tokens = TextField(new_tokens, self.token_indexer)
        document = self.generate_tokens(
                new_tokens=[new_tokens], metadata=[metadata], labels=None
        )
        return document
    
    def lime_tokenizer_viz(self, metadata, always_keep_mask, filtered_tokens):
        new_tokens = filtered_tokens
        new_tokens = [Token(t) for i, t in enumerate(metadata['tokens']) if t in new_tokens or always_keep_mask[i] == 1]
        new_tokens = TextField(new_tokens, self.token_indexer)
        document = self.generate_tokens(
                new_tokens=[new_tokens], metadata=[metadata], labels=None
        )
        return document
    
    def create_batchs(self, inputs, batch_size, query):
        return create_batch(inputs, batch_size, {"model":self.archive, "predictor": self.pred}, self.device, query) 
    