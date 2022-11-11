import time
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from easyConfig import setup_config
# import config as config
import utils

# Model
from network import TransferModel, Inception_resnet

# data
from data.model_dataset import ModelDataset
from torch.utils.data import DataLoader
from data.transform import to_tensor, augmenter
import pandas as pd

def create_store(store_file_list):
    for file in store_file_list:
        parent = os.path.split(file)[0]
        os.makedirs(parent, exist_ok=True)

def step_forward(model, optimizer, loader, criterion, device):
    running_loss = 0.0
    accuracy = Accuracy().to(device = device)
    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if optimizer != None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        accuracy.update(outputs, labels)
    return running_loss / len(loader), accuracy.compute().item()

def train(config):
    print('Loading ========')
    model = utils.load_template('network', config.model.name, config.model.args)
    model.to(device = config.device)

    # optimizer = optim.Adam(params = model.parameters(), **config.optim.args)
    optimizer = utils.smart_optimizer(model, 'AdamW',  **config.optim.args)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=config.n_epoch, **config.optim.scheduler)
    
    train_df = pd.read_csv(config.data.train_csv)
    val_df = pd.read_csv(config.data.val_csv)

    train_data = ModelDataset(config.data.dir_path, train_df.path, train_df.label, augmenter, config.data.scale)
    val_data = ModelDataset(config.data.dir_path, val_df.path, val_df.label, to_tensor)

    train_loader = DataLoader(train_data, batch_size = config.batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = 8, shuffle = True) # set = 8 for reduce GPU mem

    # train
    print('Training ========')
    start = time.time()
    # class_weight = torch.tensor() # imbalance dataset
    criterion = nn.CrossEntropyLoss()
    best = None
    for epoch in range(config.n_epoch):
        print("Epoch ============================ {}".format(epoch))
        model.train()
        running_loss, running_acc = step_forward(model, optimizer, train_loader, criterion, config.device)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = step_forward(model, None, val_loader, criterion, config.device)
            
            if best == None or val_accuracy >= best:
                print('Store')
                best = val_accuracy
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_accuracy': best
                }, config.store.checkpoint)
        
        scheduler.step()
        logging.info('Train loss/accuracy: {:.5f}/{:.5f}, val loss/accuracy {:.5f}/{:.5f}'.format(running_loss, running_acc, val_loss, val_accuracy))

    logging.info("Store at: {}".format(config.store.checkpoint))
    logging.info("Process time: {}".format(time.time() - start))

if __name__ == '__main__':
    # change the working dir to arg 'outputs' in command line

    config = setup_config('config', 'training', True)
    create_store(config.store.values())
    logging.basicConfig(filename=config.store.log, level=logging.INFO, filemode='w', format='%(levelname)s - %(message)s')
    utils.seed_everywhere(config.seed)
    train(config)