import json

class DefaultModel:
    
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        
    def train(self, a_venir=None):
        return None
    
    def predictor(self, inputs):
        raise NotImplementedError

    
