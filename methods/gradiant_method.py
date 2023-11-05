import torch

from utils import create_batch, summarize_attributions, get_token_offsets, show_text_with_normalized_scores_and_features
from methods.default import DefaultMethod

from captum.attr import LayerIntegratedGradients

import numpy as np
import gc
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



class GradientExplainer(DefaultMethod):
    
    def __init__(self, model, dataset, threshold, model_type="transformer_foreginer",query="What is the sentiment of this review?"):
        super(GradientExplainer, self).__init__(model, dataset, threshold,query)
        self.model_type = model_type
        self._embedding_layer = {}
        
    
    def scores(self, dataset):
        inputs = dataset["text"]
        labels = dataset["label"]
        if self.model_type == "transformer_huggingfaces":
            kwarg = {}
            kwarg['return_tensors']="pt"
            kwarg['truncation']=True
            kwarg['max_length']=512
            inputs_in, always_kp_ma, _ = self.model.create_batchs(inputs,1, self.query)
            
            self._embedding_layer = self.model._embedding_layer
            features, attentions = [], []
            
            lig = LayerIntegratedGradients(self.model.predictor, self._embedding_layer)
            offsets = get_token_offsets(inputs, self.model.tokenizer)

            attentions = []
            predicted_labels = []
            probas = []
            always_keep_masks = []
            
            i = 0
            for batch in inputs_in:
                batch = {k: v for k, v in batch.items()}
                bsl = torch.zeros(batch['input_ids'].size()).type(torch.LongTensor).to(self.model.device)
                logging.info("---------INPUT"+str(i)+"---------")

                attri,delta = lig.attribute(inputs=batch['input_ids'].to(self.model.device),
                                      baselines=bsl,
                                      additional_forward_args=(batch['attention_mask'].to(self.model.device)),
                                      n_steps = 5,
                                      target = labels[i],
                                      return_convergence_delta=True,
                                      internal_batch_size = 24
                                      )
                outputs = self.model.forward(document=batch, metadata=None, label=labels[i])
                predicted_labels.extend(outputs["predicted_labels"].detach().cpu())
                probas.extend(outputs["probs"].detach().cpu())

                attri = summarize_attributions(attri).detach().data[offsets[i]]
                attentions.append(attri.cpu().tolist())
                always_keep_mask = always_kp_ma[i][:attri.size()[0]]
                always_keep_masks.append(always_keep_mask)
                
                del outputs
                del always_keep_mask
                del attri
                torch.cuda.empty_cache()
                i += 1

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
        else :
                #init
                model = self.model.model

                _embedding_layer = [
                    x for x in list(model.modules()) if any(str(y) in str(type(x)) for y in model.embedding_layers)
                ]
                assert len(_embedding_layer) == 1

                self._embedding_layer['embedding_layer'] = _embedding_layer[0]
                del _embedding_layer
                
                
                inputs_in , always_keep_masks, kept_tokens = create_batch(inputs, 1, {"model":self.model.archive, "predictor": self.model.pred},device=self.model.device)
            
                self.model.model.eval()
                attentions = []
                features = []

                with torch.no_grad():
                    outputs = self.model.model._forward(inputs_in[0], None, label=torch.ones(1, dtype=torch.int64).to(self.model.device))
                    predicted_labels = outputs["predicted_labels"].detach().cpu()
                    probas = outputs["probs"].detach().cpu()
                    del outputs
                
                torch.cuda.empty_cache()
                gc.collect()
                with torch.enable_grad() :
                    self.model.model.train()
                    for param in self._embedding_layer['embedding_layer'].parameters():
                        param.requires_grad = True

                    embeddings_list = []
                    def forward_hook(module, inputs, output):  # pylint: disable=unused-argument
                        embeddings_list.append(output)
                        output.retain_grad()

                    hook = self._embedding_layer['embedding_layer'].register_forward_hook(forward_hook)

                    output_dict = self.model.model._forward(inputs_in[0], None, label=torch.ones(1, dtype=torch.int64).to(self.model.device).detach())

                    hook.remove()
                    assert len(embeddings_list) == 1
                    embeddings = embeddings_list[0]

                    predicted_class_probs = output_dict["probs"][
                        torch.arange(output_dict["probs"].shape[0]), output_dict["predicted_labels"]
                    ]  # (B, C)

                    torch.cuda.empty_cache()
                    predicted_class_probs.sum().backward(retain_graph=True)
                    torch.cuda.empty_cache()

                    gradients = ((embeddings * embeddings.grad).sum(-1).detach()).abs()
                    gradients = gradients / gradients.sum(-1, keepdim=True)

                    output_dict['attentions'] = gradients
                    output_dict['attentions'] = output_dict['attentions'][:, :kept_tokens[0].shape[1]] * (1 - kept_tokens[0]).float()
                    output_dict['attentions'] = output_dict['attentions'] / output_dict['attentions'].sum(-1, keepdim=True)
                    
                    attentions.append(output_dict['attentions'].detach().cpu().data.tolist())

                    del output_dict
                    del predicted_class_probs
                    del gradients
                    del kept_tokens
                    del embeddings
                    
                    torch.cuda.empty_cache()
                    gc.collect()

                attentions = [item for sublist in attentions for item in sublist]
            
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
        
        return {"features":features, "scores":attentions, "predicted_labels": predicted_labels, "probas": probas, "always_keep_masks": always_keep_masks, "metadatas":metadatas}
            
            

    def visualize(self, dataset, top_k=None):
        
        score_dict = self.scores(dataset=dataset)
        
        for i in range(len(score_dict["scores"])):
            saliency = score_dict["scores"][i] + score_dict["metadatas"][i]["always_keep_mask"] * -10000
            saliency = saliency.tolist() + [0.0] * (len(score_dict["metadatas"][i]["all_tokens"]) - len(saliency))
            saliency = [0.0 if e < 0.0 else e for e in saliency ]
            print(saliency, max(saliency))

            pos = np.array(saliency) > 0.0
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
