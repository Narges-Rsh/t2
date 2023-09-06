import torch
import torch.nn as nn
from .ModelBase import ModelBase
import torchmetrics

class AggModel(ModelBase):
    def __init__(self, classes, model1, model2, model3, model4, model5, model6):
        self.save_hyperparameters(
            ignore=['model1', 'model2', 'model3', 'model4', 'model5', 'model6']
        )
        super().__init__(classes)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.metrics = {'F1': torchmetrics.classification.MulticlassF1Score(num_classes=len(classes)),
        }
        self.metrics = torchmetrics.MetricCollection(self.metrics)
        self.c=0
        self.d=0
        self.p0=0
        self.p1=0
        self.p2=0
        self.p3=0
        self.p4=0
        self.p5=0
    def forward(self, x1, x2, x3,x4, x5, x6, target):
        models=[self.model1, self.model2, self.model3, self.model4, self.model5, self.model6]

        for model in models:
            model.eval()
        
        with torch.no_grad():
            #get output and probabilities from each model
            outputs=torch.stack([model(x)for model, x in zip(models, [x1,x2,x3,x4,x5,x6])])
            probabilities=torch.softmax(outputs, dim=2)
            
            predictions = [torch.argmax(probabilities[i, :, :], dim=1) for i in range(6)]
            #print("predictions", predictions)
        
            #voting part:
            combined_probabilities = torch.sum(probabilities, dim=0)
            voting_model = torch.argmax(combined_probabilities, dim=1)
            #print('predicted:',voting_model)
        

            self.metrics.update(voting_model, target)

            # Count the correct predictions
            correct_predictions = (voting_model == target).sum().item()
            self.c += correct_predictions

            incorrect_predictions = (voting_model != target).sum().item()
            self.d += incorrect_predictions
            
            correct_predictions_models = [(predictions[i] == target).sum().item() for i in range(6)]
            
            self.p0 += correct_predictions_models[0]
            self.p1 += correct_predictions_models[1]
            self.p2 += correct_predictions_models[2]
            self.p3 += correct_predictions_models[3]
            self.p4 += correct_predictions_models[4]
            self.p5 += correct_predictions_models[5]


        return self.c, self.d, self.p0,self.p1, self.p2, self.p3, self.p4, self.p5
            
    