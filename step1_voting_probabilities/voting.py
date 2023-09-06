from models import *
from models import DeepFIR
import torch
import pytorch_lightning as pl
from data.Channel_DataModule import Channel_DataModule
from models import DeepFIR
import os
import numpy as np
from models import AggModel
import matplotlib.pyplot as plt


#loading 6 pre-trained models
checkpoint_paths=[ 
       "/home/nrashvan/new_dataset_August_22/retrain_channel_0/logs/DeepFIR-Channel_DataModule/version_3/checkpoints/epoch=18-step=11400.ckpt",
       "/home/nrashvan/new_dataset_August_22/channel_1/logs/DeepFIR-Channel_DataModule/version_0/checkpoints/epoch=11-step=56256.ckpt",
       "/home/nrashvan/new_dataset_August_22/retrain_chnnel_2/logs/DeepFIR-Channel_DataModule/version_1/checkpoints/epoch=14-step=9000.ckpt",
       "/home/nrashvan/new_dataset_August_22/channel_3/logs/DeepFIR-Channel_DataModule/version_0/checkpoints/epoch=11-step=56256.ckpt",
      "/home/nrashvan/new_dataset_August_22/channel_4/logs/DeepFIR-Channel_DataModule/version_0/checkpoints/epoch=18-step=89072.ckpt",
      "/home/nrashvan/new_dataset_August_22/channel_5/logs/DeepFIR-Channel_DataModule/version_0/checkpoints/epoch=18-step=89072.ckpt"
] 

models=[DeepFIR.load_from_checkpoint(checkpoint_path, map_location= 'cuda:0') for checkpoint_path in checkpoint_paths]


dm_list= [
       Channel_DataModule(f'/home/nrashvan/new_dataset_August_22/split_data_each_channel_whole_data/channel_{i}.h5py', batch_size=1)
       for i in range (6)
       ]
#implementing step_1 for 1 million data points
for dm in dm_list:
       dm.setup(limit_samples=1000000)

pl.seed_everything(42)
c=0

#AggModel: an aggregation model in the central node based on the summation of the probabilities from 6 pre-trained models
model = AggModel(dm_list[0].classes, *models)
model.to('cuda:0')

for batch_0, batch_1, batch_2, batch_3, batch_4, batch_5  in zip(*[dm.test_dataloader() for dm in dm_list]):
        c+=1

        x_0, y_0, snr_0 = batch_0
        x_1, y_1, snr_1 = batch_1
        x_2, y_2, snr_2 = batch_2
        x_3, y_3, snr_3 = batch_3
        x_4, y_4, snr_4 = batch_4
        x_5, y_5, snr_5 = batch_5

        
        x_0 = x_0.to('cuda:0')
        x_1 = x_1.to('cuda:0')
        x_2 = x_2.to('cuda:0')
        x_3= x_3.to('cuda:0')
        x_4 = x_4.to('cuda:0')
        x_5 = x_5.to('cuda:0')
        y_0 = y_0.to('cuda:0')
        

        model.eval()
        with torch.no_grad():
             p,d, p0, p1, p2, p3,p4 ,p5= model.forward(x_0, x_1, x_2,x_3,x_4, x_5, y_0)
       
metrics_dict = model.metrics.compute()
print(metrics_dict)

# Get the F1 score
f1_score = metrics_dict.get('F1', 0.0)  

# Plot the F1 score 
plt.figure()
plt.scatter(['Test Set'], [f1_score.cpu()], color='blue', marker='o', label=f'F1 Score: {f1_score:.4f}')
plt.ylabel('F1 Score')
plt.title('F1 Score on Test Set')
plt.ylim(0, 1)  
plt.legend()
plt.savefig('f1_score_figure.png')

#print the number of correct predictions from central node and each model
print("number of test samples:", c)
print("number of total correct predictions", p)
print("number of total incorrect predictions", d)
correction_prediction_list = [p0, p1, p2, p3, p4, p5]
for i, p in enumerate(correction_prediction_list):
    print(f"number of correct predictions_{i}", p)











