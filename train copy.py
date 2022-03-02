import time
import torch
import torch.nn as nn
import torch.optim as optim
import config as cf
import utils
from tqdm.auto import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDL
from torch.util.data.distributed import DistributedSampler
from torch.util.data import DataLoader

import argparse
import os
import random
import pandas as pd
import numpy as np

# data
from data.model_dataset import ModelDataset
from data.transform import to_tensor, augmenter

# Model
from network import TransferModel, Inception_resnet

def set_random_seed(seed = 0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(gpu, args):
    print("Training GPU: ", gpu)
    
    # init process
    rank = args.node_rank * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        world_size = args.world_size,
        rank = rank
    )

    # set device & seed
    device = torch.device(f'cuda:{gpu}')
    set_random_seed(args.seed)
    torch.set_device(device)
    
    model = TransferModel(cf.model['n_class'], cf.model['backbone'], cf.model['drop'])
    model.to(device = device)
    ddl_model = DDL(model, device_ids = [gpu], output_device = gpu)

    if cf.model['freeze']:
        param_dict = [
            {'params': ddl_model.head_params}
        ]
    else:
        param_dict = [
            {'params': ddl_model.backbone_params, 'lr': cf.optim['lr'] * 0.1},
            {'params': ddl_model.head_params}
        ]

    optimizer = optim.Adam(params = param_dict, lr = cf.optim['lr'], weight_decay = 0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = cf.optim['step'], gamma=cf.optim['reduce_rate'])


    train_df = pd.read_csv(cf.data['train_csv'])
    val_df = pd.read_csv(cf.data['val_csv'])

    train_data = ModelDataset(cf.data['dir_path'], train_df.path, train_df.label, augmenter)
    val_data = ModelDataset(cf.data['dir_path'], val_df.path, val_df.label, to_tensor)
    train_sampler = DistributedSampler(train_data, num_replicas = args.world_size, rank=rank)

    train_loader = DataLoader(train_data, 
                            batch_size = cf.optim['batch_size'], 
                            sampler = train_sampler, 
                            shuffle = False, 
                            pin_memory = True,
                            num_workers = 2,
                            drop_last = True)
    val_loader = DataLoader(val_data, batch_size = 8, shuffle = False, num_workers = 1)

    # train
    if rank == 0:
        start = time.time()

    # class_weight = torch.tensor()
    criterion = nn.CrossEntropyLoss()
    best = -1
    if cf.optim['continue_training'] and os.path.exists(cf.model['checkpoint']):
        checkpoint = torch.load(cf.model['checkpoint'], map_location = device)
        ddl_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best = checkpoint['val_accuracy']

    for epoch in range(cf.optim['n_epoch']):
        print("Epoch ============================ {}".format(epoch))
        train_sampler.set_epoch(epoch)
        ddl_model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(cf.device), labels.to(cf.device)

            optimizer.zero_grad()
            outputs = ddl_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        running_loss /= len(train_loader)
        if rank == 0:
            ddl_model.eval()
            with torch.no_grad():
                val_loss, val_accuracy = 0.0, 0.0
                for inputs, labels in val_loader:
                    inputs = inputs.to(device = cf.device)
                    outputs = ddl_model(inputs)
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
                        'model': ddl_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_accuracy': best
                    }, cf.model['checkpoint'])
        
        scheduler.step()
        print('Train loss: {}, val loss {}, val accuracy {}'.format(running_loss, val_loss, val_accuracy))

    if rank == 0:    
        print("Store at: ", cf.model['checkpoint'])
        print("Process time: ", time.time() - start)
