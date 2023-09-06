#central node based on a fully connected neural network
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import torchmetrics

# load the file containing concatenated features 
input_file = "/home/nrashvan/new_dataset_August_22/Fed_AMR/step2_feature_extraction/concatenated_features.npz"
loaded_data = np.load(input_file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Keys in the loaded data:", list(loaded_data.keys()))

features = loaded_data['features']
labels_0 = loaded_data['labels_0']
labels_1 = loaded_data['labels_1']
labels_2 = loaded_data['labels_2']
labels_3 = loaded_data['labels_3']
labels_4 = loaded_data['labels_4']
labels_5 = loaded_data['labels_5']
snr_0 = loaded_data['snr_0']
snr_1 = loaded_data['snr_1']
snr_2 = loaded_data['snr_2']
snr_3 = loaded_data['snr_3']
snr_4 = loaded_data['snr_4']
snr_5 = loaded_data['snr_5']

print("Shape of features:", features.shape)


#train (60%), validation (20%), and test (20%) sets
x_train, x_temp, y_train, y_temp = train_test_split(features, labels_0, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)


x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)  
x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

batch_size=512
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size)

print("x_train.shape", x_train.shape)


# a fully connected neural network model with three hidden layers for the central node

class classifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob=0.3):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 2000) 
        self.fc3 = nn.Linear(2000, 500)   
        self.fc4 = nn.Linear(500, num_classes)  
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  
        return x
        

epochs=4
model = classifier(input_size=x_train.shape[1], num_classes=8).to(device)
model.to(device)
print("number of samples for training", x_train.shape[0])
print("input size", x_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

f1_sc = torchmetrics.classification.MulticlassF1Score(num_classes=8).to(device)

train_losses = []
val_losses = []
val_f1_scores=[]

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
      
      
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() 
         
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    
    with torch.no_grad():
        all_val_predicted_labels = []
        all_val_actual_labels = []
        val_loss = 0.0
        
        for batch_features, batch_labels in val_loader:
            outputs = model(batch_features)
            val_loss += criterion(outputs, batch_labels).item()

            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            actual = batch_labels.cpu().numpy()

            all_val_predicted_labels.extend(predicted)
            all_val_actual_labels.extend(actual)
            f1_sc.update(outputs.data, batch_labels)
            

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_f1= f1_sc.compute()
        val_f1_scores.append(val_f1)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f},F1 Score (Validation): {val_f1:.4f}")
        f1_sc.reset()

model_filename = f"Centralized_trained_model.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved to '{model_filename}'")

# Plot confusion_matrix_validation 
class_names= ['bpsk', 'qpsk', '8psk', 'dqpsk', 'msk', '16qam', '64qam', '256qam']  
val_conf_matrix = confusion_matrix(all_val_actual_labels, all_val_predicted_labels)
plt.figure(figsize=(8, 6))
plt.imshow(val_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Validation Set)")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(f"confusion_matrix_validation'.png")

# Plot validation and train loss 
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"loss_plot.png") 

# Plot validation F1 score 
val_f1_scores_cpu = [val_f1.cpu().item() for val_f1 in val_f1_scores] 

plt.figure()
plt.plot(range(1, epochs+1), val_f1_scores_cpu, label='Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig(f'F1Score_plot_validation.png') 

# Test loop
model.eval()
all_predicted_labels = []
all_actual_labels = []
test_f1_scores=[]
with torch.no_grad():
    
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features)
        _, predicted = torch.max(outputs.data, 1)
        
        predicted = predicted.cpu().numpy()
        actual = batch_labels.cpu().numpy()

        all_predicted_labels.extend(predicted)
        all_actual_labels.extend(actual)
        f1_sc.update(outputs.data, batch_labels)


# Plot the F1 score for the test set 
test_f1 = f1_sc.compute()
test_f1_scores.append(test_f1)
print(f"F1 Score (Test): {test_f1:.4f}")
f1_sc.reset()

test_f1_cpu = test_f1.cpu().item() 

plt.figure()
plt.scatter(['Test Set'], [test_f1_cpu ], color='blue', marker='o', label=f'F1 Score: {test_f1_cpu:.4f}')
plt.ylabel('F1 Score')
plt.title('F1 Score on Test Set')
plt.ylim(0, 1) 
plt.legend()
plt.savefig(f'f1_score_plot.png')


#confusion_matrix for test set
conf_matrix = confusion_matrix(all_actual_labels, all_predicted_labels)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Test Set)")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks,  class_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(f'confusion_matrix_test.png')


# see some outputs of the model
model.eval()

num_samples_to_inspect = 2
with torch.no_grad():
    for i, (batch_features, batch_labels) in enumerate(test_loader):
        if i >= num_samples_to_inspect:
            break
        outputs = model(batch_features)
         
        print(f"Sample {i+1}:")
        print("Raw Output Scores (Logits):")
        print(outputs)

        print(f"Sample {i+1}:")
        _, predicted = torch.max(outputs.data, 1)

        predicted = predicted.cpu().numpy()
        actual = batch_labels.cpu().numpy()

       
        print(f"Sample {i+1}:")
        print("Predicted Labels:", predicted)
        print("Actual Labels:", actual)
        print("=" * 10)
