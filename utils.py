from torch.utils.data import Dataset
import numpy as np
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data import Instance
from allennlp.data.dataset import Batch
import json
import torch
from IPython.display import display, HTML
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(list(self.data_dict.values())[0])  # Nombre d'exemples, ici on suppose que tous les tensors ont la même taille

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data_dict.items()}

def create_batches(inputs, batch_size):
    num_batches = len(inputs) // batch_size
    
    if num_batches!=0:
        # Divise la liste des inputs en lots
        batches = np.array_split(inputs, num_batches)
    else:
        batches = np.array([inputs])
    return batches

def create_dataset(inputs, inp_stream, device, query=None):
    tokens=[]
    always_keep_masks = []
    for docwords in inputs:
        token=[]
        document_tokens = [Token(word) for word in docwords.split()]
        token += document_tokens
        
        always_keep_mask = [0] * len(document_tokens)
        always_keep_mask += [1]

        token.append(Token("[SEP]"))
        
        if query != None:
            token += [Token(word) for word in query.split()]
            always_keep_mask += [1] * (len(query.split()) + 1)
            token.append(Token("[SEP]"))
            
        always_keep_masks.append(always_keep_mask)
        
        
        tokens.append(token)

    instances = []
    for token in tokens:
        instances += [Instance({"document": TextField(token, inp_stream["predictor"]._dataset_reader._token_indexers)})]

    batch = Batch(instances)
    batch.index_instances(inp_stream["model"].model._vocabulary)
    padding_lengths = batch.get_padding_lengths()
    batch = batch.as_tensor_dict(padding_lengths)

    return {k: v.to(device) for k, v in batch["document"].items()}, always_keep_masks

def create_dataset_with_mask(inputs, inp_stream, device, query):
    torch.cuda.empty_cache()
    tokens=[]
    always_keep_masks = []
    for docwords in inputs:
        token=[]
        document_tokens = [Token(word) for word in docwords.strip().split(' ')]
        token += document_tokens
        
        always_keep_mask = [0] * len(document_tokens)
        always_keep_mask += [1]

        token.append(Token("[SEP]"))
        
        if query != None:
            token += [Token(word) for word in query.split()]
            always_keep_mask += [1] * (len(query.split()) + 1)
            token.append(Token("[SEP]"))
        
        always_keep_masks.append(always_keep_mask)
        
        
        tokens.append(token)

    instances = []
    kept_tokens = []
    i=0
    for token in tokens:
        doc = TextField(token, inp_stream["predictor"]._dataset_reader._token_indexers)
        instances += [Instance({"document": doc})]
        kept_tokens += [Instance({"kept_tokens": SequenceLabelField(always_keep_masks[i], sequence_field=doc, label_namespace="kept_token_labels")})]
        i+=1

    batch = create_batch_from_instances(instances, inp_stream["model"].model._vocabulary)
    batch_kpt = create_batch_from_instances(kept_tokens, inp_stream["model"].model._vocabulary)
    del doc 
    del tokens
    del instances
    del kept_tokens
    torch.cuda.empty_cache()

    return {k: v.to(device).detach() for k, v in batch["document"].items()}, always_keep_masks, batch_kpt["kept_tokens"].to(device).detach()

def create_batch(inputs, batch_size, inp_stream, device, query):
    outputs = []
    always_keep_masks = []
    kept_tokens = []
    num_batches = len(inputs) // batch_size
    
    if num_batches!=0:
        # Divise la liste des inputs en lots
        batches = np.array_split(inputs, num_batches)
    else:
        batches = np.array([inputs])
        
    for inp in batches:
        dataset = create_dataset_with_mask(inp, inp_stream, device, query)
        outputs.append(dataset[0])
        always_keep_masks.extend(dataset[1])
        kept_tokens.append(dataset[2])
        del dataset
        
    return outputs, always_keep_masks, kept_tokens

def create_batch_from_instances(instances, vocab):
    batch = Batch(instances)
    batch.index_instances(vocab)
    padding_lengths = batch.get_padding_lengths()
    batch = batch.as_tensor_dict(padding_lengths)
    return batch


def create_dataset_huggingface(inputs, inp_stream):
    tokenizer = inp_stream["tokenizer"]
    return tokenizer(inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=512)     

def save_to_json_file(file_name, dicti):
    with open(file_name, 'w') as json_file:
        json.dump(dicti, json_file)
        
    print(f"Le dictionnaire a été sauvegardé dans le fichier {file_name}")
    
def create_always_keep_mask(inputs, query):
    always_keep_masks = []
    kept_tokens = []
    for docwords in inputs:
        
        always_keep_mask = [0] * len(docwords.split())
        always_keep_mask += [1]        
        
        if query != None:
            always_keep_mask += [1] * (len(query.split()) + 1)
        
        always_keep_masks.append(always_keep_mask)
        kept_tokens.append(torch.tensor(always_keep_mask).detach())

    return always_keep_masks, kept_tokens


def create_batch_huggingface(inputs, query, tokenizer, device, kwarg):
    always_keep_masks = []
    kept_tokens = []
    inputs_in = []
    for docwords in inputs:        
        always_keep_mask = [0] * len(list(filter(lambda x: bool(len(x)), docwords.strip().split(' '))))
        always_keep_mask += [1]        
        
        if query != None:
            always_keep_mask += [1] * (len(query.split()) + 1)
            kwarg['text'] = " ".join([docwords, "[SEP]", query, "[SEP]"])
        else:
            kwarg['text'] = " ".join([docwords, "[SEP]"])
        
        always_keep_masks.append(always_keep_mask)
        kept_tokens.append(torch.tensor([always_keep_mask]).to(device).detach())
        
        inputs_in.append({k: v.to(device) for k, v in tokenizer(**kwarg).items()})
        

    return inputs_in, always_keep_masks, kept_tokens

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def summarize_attentions(attentions):
    attention_tensor = attentions.squeeze(0)
    aggregated_attention = attention_tensor.mean(dim=0)
    normalized_attention = aggregated_attention / aggregated_attention.sum()
    importance_features = normalized_attention
    return importance_features[:,0]

def get_token_offsets(inputs, tokenizer, query=None):

    offsets_res = []
        
    for text in inputs:

        # Utilisez la méthode encode_plus pour obtenir les offsets
        if query != None:
            t_input = " ".join([text, "[SEP]", query, "[SEP]"])
        else:
            t_input = " ".join([text, "[SEP]"])

        encoding = tokenizer.encode_plus(
            t_input,
            add_special_tokens=False,
            return_offsets_mapping=True
        )

        offsets = encoding['offset_mapping']

        # Filtrer les tokens spéciaux (CLS, SEP, PAD, etc.)
        token_start_indices = []
        token_end_indices = []

        for i, (start, end) in enumerate(offsets):
            if start is not None and end is not None:
                token_start_indices.append(start)
                token_end_indices.append(end-1)

        # Convertir les indices de caractères en indices de début de mots par rapport aux tokens
        word_start_indices = [0]

        for i in range(1, len(token_start_indices)):
            # Vérifiez si le token actuel commence par un espace (nouveau mot)
            if t_input[token_start_indices[i]-1] == " " and t_input[token_end_indices[i]] != " ":
                word_start_indices.append(i)


        offset = np.array(word_start_indices)
        offsets_res.append(offset[offset<512])

    return offsets_res

def save_html_to_file(html_content, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(html_content)

def show_text_with_normalized_scores_and_features(text, token_scores, top_k_features, top_k_scores, predicted_class, norm=1):
    normalized_scores = np.array(token_scores)
    normalized_scores = (normalized_scores - np.min(normalized_scores)) / (np.max(normalized_scores) - np.min(normalized_scores))
    
    top_k_scores = (top_k_scores - np.min(top_k_scores)) / (np.max(top_k_scores) - np.min(top_k_scores))
    
    html = ""
    
    for token, score in zip(text, normalized_scores):
        html += f'<span style="background-color: rgba(255, 140, 0, {score});">{token} </span>'
    
    top_k_features_str = ', '.join(top_k_features)
    html += f'<br><br><strong>Top {len(top_k_features)} features:</strong><br>'
    
    # Créer un DataFrame pandas pour afficher les top_k features et scores
    features_df = pd.DataFrame({'Feature': top_k_features, 'Score': top_k_scores})
    html += features_df.to_html(index=False)
    
    html += f'<br><strong>Predicted class:</strong> {predicted_class}'

    save_html_to_file(html, 'output_'+str(norm)+'.html')
    
    display(HTML(html))

def f1(_p, _r):
    if _p == 0 or _r == 0:
        return 0
    return 2 * _p * _r / (_p + _r)

def features_accuracy(gold_list, features, k_features=10):
        
        num_elements = (len(features) * k_features)//100
        if num_elements == 0:
            accuracy, recall = 0,0
        else:
            common_features = set(gold_list) & set(features[:num_elements])

            # Calcul de l'accuracy + recall
            accuracy = len(common_features) / len(gold_list)
            recall = len(common_features) / num_elements

        return accuracy, recall, f1(accuracy, recall)
    
def verify_dataset(dataset):
    infos = dataset.info

    # Récupérer le premier attribut
    attribut = list(infos.features.keys())

    if len(attribut) < 2:
        print("Wrong format dataset need to contain at least lebel and text")
        exit(1)
    
    boole = 'evidences' in attribut
    
    if attribut[1] == 'text':
        return boole, dataset
    else:
        return boole, dataset.rename_column(attribut[0],"text")
    
def verify_tokenizer(home, url):
    response = False
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    try:
        tokenizer = AutoTokenizer.from_pretrained(home+"/"+url)
        response=True
        model_type = "transformer_huggingfaces"
    except Exception as e:
        pass
    
    if not response:
        try:
            tokenizer = AutoTokenizer.from_pretrained(url)
        except Exception as e:
            pass
        
    return tokenizer

def calculate_comp_value(out_dict):
    value = {}
    
    #sufficiency
    aopc_thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    aopc = []
    for t in aopc_thresholds:
        aopc.append(np.average(np.array(out_dict['original_probs']) - np.array(out_dict['sufficiency_probs'][t])))
    
    value['sufficiency'] = aopc
    value['aopc_sufficiency'] = np.average(aopc)
    
    #comprehensiveness
    aopc = []
    for t in aopc_thresholds:
        aopc.append(np.average(np.array(out_dict['original_probs']) - np.array(out_dict['comprehensiveness_probs'][t])))
    
    value['comprehensiveness'] = aopc
    value['aopc_comprehensiveness'] = np.average(aopc)
    save_to_json_file("comprehensibility_metrics.json", value)
