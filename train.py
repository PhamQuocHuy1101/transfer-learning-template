import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import config as cf

# Model
from network import Model

model = Model(cf.model['n_class'], cf.model['backbone'], cf.model['drop'])
model.to(device = cf.device)
if cf.model['freeze']:
    if 'resnet' in cf.model['backbone']:
        param_dict = [
            {'params': [p for n, p in model.named_parameters() if 'fc' in n]},  
        ]
    if 'b' in cf.model['backbone']:
        param_dict = [
            {'params': [p for n, p in model.named_parameters() if 'classifier' in n]},
        ]
else:
    param_dict = [{'params': model.parameters()}]

optimizer = optim.Adam(params = param_dict, lr = cf.optim['lr'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = cf.optim['step'], gamma=cf.optim['reduce_rate'])

# data
from data.model_dataset import ModelDataset
from torch.utils.data import DataLoader
from data.transform import to_tensor, augmenter
import pandas as pd

train_df = pd.read_csv(cf.data['train_csv'])
val_df = pd.read_csv(cf.data['val_csv'])

train_data = ModelDataset(cf.data['dir_path'], train_df.path, train_df.label, augmenter)
val_data = ModelDataset(cf.data['dir_path'], val_df.path, val_df.label, to_tensor)

train_loader = DataLoader(train_data, batch_size = cf.optim['batch_size'], shuffle = True)
val_loader = DataLoader(val_data, batch_size = 8, shuffle = True)

# train
start = time.time()
from tqdm.auto import tqdm

# class_weight = torch.tensor()
criterion = nn.CrossEntropyLoss()
best = -1
if cf.optim['continue_training'] and os.path.exists(cf.model['checkpoint']):
    checkpoint = torch.load(cf.model['checkpoint'], map_location = cf.device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best = checkpoint['val_accuracy']

for epoch in range(cf.optim['n_epoch']):
    print("Epoch ============================ {}".format(epoch))
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(cf.device), labels.to(cf.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    running_loss /= len(train_loader)
    model.eval()
    with torch.no_grad():
        val_loss, val_accuracy = 0.0, 0.0
        for inputs, labels in val_loader:
            inputs = inputs.to(device = cf.device)
            outputs = model(inputs)
            outputs = outputs.cpu()
            val_loss += criterion(outputs, labels).item()
            y_predict = torch.softmax(outputs, dim = 1).argmax(dim = 1)
            val_accuracy += torch.sum(y_predict == labels).item()
        val_loss /= len(val_loader)
        val_accuracy /= len(val_data)
        if best == -1 or val_accuracy >= best:
            print('Store')
            best = val_accuracy
            torch.save({
                'model': model.state_dict(),
                'optimizer': model.state_dict(),
                'val_accuracy': best
            }, cf.model['checkpoint'])
    
    scheduler.step()
    print('Train loss: {}, val loss {}, val accuracy {}'.format(running_loss, val_loss, val_accuracy))

print("Store at: ", cf.model['checkpoint'])
print("Process time: ", time.time() - start)
