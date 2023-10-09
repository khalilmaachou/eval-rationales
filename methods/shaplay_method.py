import json
import torch
import shap

class ShaplayExplainer(DefaultMethod):
    
    def __init__(self, model, dataset):
        super(ShaplayExplainer, self).__init__(model, dataset)
    
    def scores(self, inputs):
        exp = shap.Explainer(self.model.predictor, output_names=['negative', 'positive'])
        shap_values = exp(self.dataset['test']['review'])
        features, attentions = [], []
        
        i=0
        for row in self.dataset['test']:
            predicted_label = self.model.predictor([row['review']])
            valeur_max = max(predicted_label[0])
            predicted_label = predicted_label[0].tolist().index(valeur_max)

            saliency=[]
            weights=shap_values.values[i][:,predicted_label]
            for w in weights:
                saliency.append(max(0.0, w))
            
            #Rank the features from the best to the worst
            args = np.argsort(saliency)[::-1]
            
            loc_features = (shap_values.data[i][args])[np.array(saliency)!=0] 
            
            #Calculate the attention 
            saliency = torch.Tensor([saliency])
            attention = saliency
            if(len(loc_features)!=0):
                attention = attention / attention.sum(-1, keepdim=True)
            
            attentions.append(attention)
            features.append(loc_features)
            i+=1
        
        return {"features":features, "scores":attentions}

    
