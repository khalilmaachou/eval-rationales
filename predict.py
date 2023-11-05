# -*- coding: utf-8 -*-
import json
import os
import sys
from datasets import load_dataset, load_from_disk, DatasetDict
import numpy as np
import logging
import torch
from pathlib import Path


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

home_directory = os.path.expanduser("~")

sys.path.append(home_directory+"/explainer")
sys.path.append(home_directory+"/Eraser-Benchmark-Baseline-Models")

from Rationale_model.models.classifiers.soft_encoder_model import SoftEncoderRationaleModel
import Rationale_model.saliency_scorer.lime
import Rationale_model.data.dataset_readers.rationale_reader

from model.transformer_foreginer import Foreginer_Transformer
from model.transformer_huggingfaces import HuggingFace_Transformer
from model.linear_regressor import Linear_Model
from model.decision_tree import Tree_Model

from methods.attention_method import AttentionExplainer
from methods.lime_method import LimeExplainer
from methods.gradiant_method import GradientExplainer
from methods.random import RandomExplainer


from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoModel, AutoConfig


from utils import save_to_json_file, verify_dataset, verify_tokenizer, calculate_comp_value

home_directory = os.path.expanduser("~")

sys.path.append(home_directory+"/explainer")

from methods.lime_method import LimeExplainer
import argparse
    

if __name__ == "__main__":

    # Créer un objet ArgumentParser
    parser = argparse.ArgumentParser(description='Script pour effectuer des opérations avec des paramètres.')

    # Ajouter les arguments
    parser.add_argument('--data', type=str, required=True, help='URL des données')
    parser.add_argument('--model', type=str, required=True, help='URL du modèle')
    parser.add_argument('--saliency', type=str, required=True, help='Type de saliency')
    parser.add_argument('--task', type=str)
    parser.add_argument('--mimic_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--predictor_type', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--metrics', type=str)
    parser.add_argument('--visualize', type=str)
    parser.add_argument('--query', type=str)

    # Analyser les arguments de la ligne de commande
    args = parser.parse_args()

    # Accéder aux valeurs des arguments
    url_data = home_directory+"/"+args.data
    url_model = home_directory+"/"+args.model
    saliency = args.saliency
    task = args.task
    mimic_path = args.mimic_path
    model_type = args.model_type
    predictor_type = args.predictor_type
    split = args.split
    metrics = args.metrics
    visualize = args.visualize
    query = args.query

    # Utilisation des valeurs des arguments
    logging.info('Loading data : '+url_data)
    logging.info('Loading model : '+url_model)
    logging.info('Explaining using : '+saliency)
    
    ################ Load Data #################################
    chemin = Path(url_data)
    data_exist = False
    
    if chemin.exists():
        try:
            dataset = load_from_disk(url_data)
            data_exist=True
            
        except Exception as e:
            pass 
    elif task != None:
        try:
            dataset = load_dataset(args.data, task=task, mimic_path=mimic_path)
            data_exist=True
            
        except Exception as e:
            pass 
    else:
        try:
            dataset = load_dataset(args.data)
            data_exist=True
            
        except Exception as e:
            pass 
    
    if (not data_exist):
        print("Wrong Data URL : This dataset doesn't exist")
        exit(1)  
    
    if isinstance(dataset, DatasetDict):
        if split != None:
            dataset = dataset[split]
        else:
            dataset = dataset['test']
            
    has_evidence, dataset = verify_dataset(dataset)
    
    response = False
    try:
        MODEL = AutoModelForSequenceClassification.from_pretrained(args.model, output_attentions=True)
        response=True
        model_type = "transformer_huggingfaces"
    except Exception as e:
        pass
    
    if not response:
        try:
            MODEL = AutoModelForSequenceClassification.from_pretrained(url_model, output_attentions=True)
            response=True
            model_type = "transformer_huggingfaces"
        except Exception as e:
            pass    
    
    if (not response) and ('.tar.gz' in url_model):
        model_type = "transformer_foreginer"
        response=True
        
    if not response:
        try:
            with open(url_model, 'rb') as fichier:
                modele_charge = pickle.load(fichier)
            
            if isinstance(modele_charge, DecisionTreeClassifier):
                model_type = "pre_trained_decision_tree"
            elif isinstance(modele_charge, LogisticRegression):
                model_type = "pre_trained_linear_regressor"
            
            response=True
        except Exception as e:
            pass        
        
        
    if (not response) and (model_type == None):
        print("Wrong Model URL : This model doesn't exist")
        # Arrêtez le script ici si nécessaire
        exit(1)    

    
    if model_type == "transformer_huggingfaces":
        att = str(MODEL).split("(")[2].split(")")[0]
        attribut_model = getattr(MODEL, att)
        tokenizer = verify_tokenizer(home_directory, args.model)
        model = HuggingFace_Transformer(dataset, model=MODEL, embedding_layer=attribut_model.embeddings, tokenizer=tokenizer)
    elif model_type == "transformer_foreginer":
        model = Foreginer_Transformer(url_model,predictor_type, dataset)
    elif model_type == "pre_trained_decision_tree":
        model = Tree_Model(dataset)
        model.model = modele_charge
    elif model_type == "pre_trained_linear_regressor":
        model = Linear_Model(dataset)
        model.model = modele_charge
    elif model_type == "decision_tree":
        model = Tree_Model(dataset)
    elif model_type == "linear_regressor":
        model = Linear_Model(dataset)
        
        
    
    if saliency == "lime":
        saliency_scorer = LimeExplainer(model, dataset, 0.5, model_type=model_type, query=query)
    elif saliency == "attention":
        saliency_scorer = AttentionExplainer(model, dataset, 0.5, model_type=model_type, query=query)
    elif saliency == "gradient":
        saliency_scorer = GradientExplainer(model, dataset, 0.5, model_type=model_type, query=query)
    elif saliency == "random":
        saliency_scorer = RandomExplainer(model, dataset, 0.5, model_type=model_type, query=query)
    elif saliency == "white":
        pass
    
    if metrics != None:
        scorer_dict = saliency_scorer.scores(dataset[:3])
    
        out_dict = saliency_scorer.generate_comprehessiveness_metrics(scorer_dict, dataset[:3])

        calculate_comp_value(out_dict)
        if has_evidence:
            saliency_scorer.generate_token_metrics(scorer_dict, dataset[:3])
    elif visualize:
        saliency_scorer.visualize(dataset[:2])
