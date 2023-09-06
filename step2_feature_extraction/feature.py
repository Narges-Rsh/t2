#This code extracts features using the updated DeepFIR forward path 
#for the six clients and saves the features in separate files for each client.

from models import *
import torch
from data.Load_DataModule import Load_DataModule
from models import DeepFIR
import os
import numpy as np



#loading 6 pre-trained models
checkpoint_paths = [
    "/home/nrashvan/new_dataset_August_22/retrain_channel_0/logs/DeepFIR-Channel_DataModule/version_3/checkpoints/epoch=18-step=11400.ckpt"
    "/home/nrashvan/new_dataset_August_22/channel_1/logs/DeepFIR-Channel_DataModule/version_0/checkpoints/epoch=11-step=56256.ckpt"
    "/home/nrashvan/new_dataset_August_22/retrain_chnnel_2/logs/DeepFIR-Channel_DataModule/version_1/checkpoints/epoch=14-step=9000.ckpt"
    "/home/nrashvan/new_dataset_August_22/channel_3/logs/DeepFIR-Channel_DataModule/version_0/checkpoints/epoch=11-step=56256.ckpt"
    "/home/nrashvan/new_dataset_August_22/channel_4/logs/DeepFIR-Channel_DataModule/version_0/checkpoints/epoch=18-step=89072.ckpt"
    "/home/nrashvan/new_dataset_August_22/channel_5/logs/DeepFIR-Channel_DataModule/version_0/checkpoints/epoch=18-step=89072.ckpt"]

models = []
map_location='cuda:2'
for path in checkpoint_paths:
    model = DeepFIR.load_from_checkpoint(path, map_location=map_location)
    models.append(model)



dm_list= [
       Load_DataModule(f'/home/nrashvan/new_dataset_August_22/split_data_each_channel_whole_data/channel_{i}.h5py', batch_size=1)
       for i in range (6)
       ]

for dm in dm_list:
       dm.setup(limit_samples=1000000)

pl.seed_everything(42)

#for model in models:
#    model.to(map_location)

data_modules = [dm_0, dm_1,dm_2, dm_3, dm_4, dm_5 ] 
models = [model_0, model_1, model_2, model_3, model_4, model_5]

# Loop through each data module
for dm_index, dm in enumerate(data_modules):
    
    model = models[dm_index]  

    extracted_features = []
    corresponding_labels = []
    corresponding_snr = []

    for batch in dm.test_dataloader():
           
            x, y, snr = batch
            x, y, snr = x.to(map_location), y.to(map_location), snr.to(map_location)

            model.eval()
            with torch.no_grad():
                features = model.forward(x)

            
            extracted_features.append(features.cpu().numpy())
            corresponding_labels.append(y.cpu().numpy())
            corresponding_snr.append(snr.cpu().numpy())

    
    data_dict = {
        'features': np.vstack(extracted_features),
        'labels': np.hstack(corresponding_labels),
        'snr': np.hstack(corresponding_snr)
    }

    
    output_file = f"extracted_data_dm_{dm_index}.npy"
    np.save(output_file, data_dict)


#combining features from six models, each providing 128 features, 
#and transforms them into a single set of 768 features.


all_features = []
all_labels = []
all_snr = []

data_module_indices = [0, 1, 2, 3,4,5]  


for dm_index in data_module_indices:
    
    saved_features_file = f"extracted_data_dm_{dm_index}.npy"
    loaded_data_dict = np.load(saved_features_file, allow_pickle=True).item()

    
    loaded_features = loaded_data_dict['features']
    loaded_labels = loaded_data_dict['labels']
    loaded_snr = loaded_data_dict['snr']

    
    all_features.append(loaded_features)
    all_labels.append(loaded_labels)
    all_snr.append(loaded_snr)


    print("before concatenation:")
    print(f"Loaded features shape (Data Module {dm_index}):", loaded_features.shape)
    
    for i in range(3):
        features = loaded_features[i]
        label = loaded_labels[i]
        snr = loaded_snr[i]

        print(f"Sample {i+1}:")
        print("Features:", features)
        print("Label:", label)
        print("SNR:", snr)
        print("=" * 10)

concatenated_features = np.concatenate(all_features, axis=1)

data_dict = {
        'features': concatenated_features,
        'labels_0': all_labels[0],
        'labels_1': all_labels[1],
        'labels_2': all_labels[2],
        'labels_3': all_labels[3],
        'labels_4': all_labels[4],
        'labels_5': all_labels[5],
        'snr_0': all_snr[0] ,
        'snr_1': all_snr[1] , 
        'snr_2': all_snr[2] ,
        'snr_3': all_snr[3] ,
        'snr_4': all_snr[4],
        'snr_5': all_snr[5]
    }

print("after concatenation:")
print("Concatenated features shape:", concatenated_features.shape)
print("Number of samples in concatenated features:", concatenated_features.shape[0]) 
print("Number of features in concatenated features:", concatenated_features.shape[1]) 

for i in range(3):
    features = concatenated_features[i]
    label_0 = data_dict['labels_0'][i]
    label_1 = data_dict['labels_1'][i]
    label_2 = data_dict['labels_2'][i]
    label_3 = data_dict['labels_3'][i]
    label_4= data_dict['labels_4'][i]
    label_5 = data_dict['labels_5'][i]
    snr_0 = data_dict['snr_0'][i]
    snr_1 = data_dict['snr_1'][i]
    snr_2 = data_dict['snr_2'][i]
    snr_3 = data_dict['snr_3'][i]
    snr_4 = data_dict['snr_4'][i]
    snr_5 = data_dict['snr_5'][i]
    
    print(f"After concatenation: Sample {i+1}:")
    print("After concatenation:Features:", features)
    print("After concatenation:Label_0:", label_0)
    print("After concatenation:Label_1:", label_1)
    print("After concatenation:Label_2:", label_2)
    print("After concatenation:Label_3:", label_3)
    print("After concatenation:Label_4:", label_4)
    print("After concatenation:Label_5:", label_5)
    print("After concatenation:SNR_0:", snr_0)
    print("After concatenation:SNR_1:", snr_1)
    print("After concatenation:SNR_2:", snr_2)
    print("After concatenation:SNR_3:", snr_3)
    print("After concatenation:SNR_4:", snr_4)
    print("After concatenation:SNR_5:", snr_5)
    print("=" * 10)


output_file = "concatenated_features.npz"
np.savez(output_file, **data_dict)
output_path = os.path.abspath(output_file)

print("Concatenated features saved to:", output_path)











        

        
                                      
     


   











