import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-base")

# For transformers v4.x+: 
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

df_train = pd.read_csv('text/csv/train_clean.csv')
df_dev = pd.read_csv('text/csv/dev_clean.csv')
df_test = pd.read_csv('text/csv/test_clean.csv')

text_train = df_train['clean_text'].values[:8]
text_dev = df_dev['clean_text'].values[:1]
text_test = df_test['clean_text'].values[:1]

features_train = np.zeros((len(text_train), 768))
features_dev = np.zeros((len(text_dev), 768))
features_test = np.zeros((len(text_test), 768))

def encode_text(text_l):
    features = np.zeros((len(text_l), 768))
    for i, text in enumerate(text_l):
        input_ids = torch.tensor([tokenizer.encode(text)])
        input_ids = input_ids[:, :256]
        print(f'Step {i + 1}/{len(text_l)}')
        with torch.no_grad():
            vec = phobert(input_ids)[1]              # Models outputs are now tuples
        features[i] = vec
    return features

features_train = encode_text(text_train).astype(np.float32)
features_dev = encode_text(text_dev).astype(np.float32)
features_test = encode_text(text_test).astype(np.float32)

## encode y
y_train = df_train["label"].values
y_dev = df_dev["label"].values
y_test = df_test["label"].values

dic_y_mapping = {n:label for n,label in
                 enumerate(np.unique(y_train))}
inverse_dic = {v:k for k,v in dic_y_mapping.items()}

y_train = np.array([inverse_dic[y] for y in y_train])[:8]
y_dev = np.array([inverse_dic[y] for y in y_dev])[:1]
y_test = np.array([inverse_dic[y] for y in y_test])[:1]

class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], torch.tensor(self.y[idx], dtype=torch.long)

train_ds = TextDataset(features_train, y_train)
dev_ds = TextDataset(features_dev, y_dev)
test_ds = TextDataset(features_test, y_test)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=2, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

class LinearClassifier(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, out_features)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.softmax(x)
        return output

def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

# Training Function
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'Device {device}')

def train(model,
          optimizer,
          criterion = nn.CrossEntropyLoss(),
          train_loader = train_loader,
          dev_loader = dev_loader,
          num_epochs = 100,
          eval_every = len(train_ds) // 2,
          file_path = "pretrained_model",
          best_dev_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    dev_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    dev_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        print(f'Training epochs {epoch + 1}/{num_epochs}')
        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)           
            labels = labels.to(device)
            output = model(features)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():                    
        # devation loop
            for features, labels in dev_loader:
                features = features.to(device) 
                labels = labels.to(device)
                output = model(features)

                loss = criterion(output, labels)
                dev_running_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy: %d %%' % (100 * correct / total))

        # evaluation
        average_train_loss = running_loss
        average_dev_loss = dev_running_loss
        train_loss_list.append(average_train_loss)
        dev_loss_list.append(average_dev_loss)
        global_steps_list.append(global_step)

        # resetting running values
        running_loss = 0.0                
        dev_running_loss = 0.0
        model.train()

        # print progress
        print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {average_train_loss}, dev Loss: {average_dev_loss}')
                  
        # checkpoint
        if best_dev_loss > average_dev_loss:
            best_dev_loss = average_dev_loss
            save_checkpoint(file_path + f'/model.pt', model, optimizer, best_dev_loss)
    
    print('Finished Training!')

model = LinearClassifier(768, 27).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model=model, optimizer=optimizer, num_epochs=1000)
