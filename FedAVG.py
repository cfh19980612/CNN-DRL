import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import random
import copy
import pandas as pd
import numpy as np
import queue
import math
import networkx as nx
import argparse
import time

from utils import progress_bar
from tqdm import tqdm, trange
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Set_dataset(dataset):
    if dataset == 'CIFAR10':
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        parser.add_argument('--epoch',default=100,type=int,help='epoch')
        args = parser.parse_args()

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='/home/ICDCS/cifar-10-batches-py/', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='/home/ICDCS/cifar-10-batches-py/', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

        return args, trainloader, testloader
    elif dataset == 'MNIST':
        parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        parser.add_argument('--epoch',default=100,type=int,help='epoch')
        args = parser.parse_args()

        # Data
        print('==> Preparing data..')
        # normalize
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        # download dataset
        trainset = torchvision.datasets.MNIST(root = "./data/",
                        transform=transform,
                        train = True,
                        download = True)
        # load dataset with batch=64
        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                            batch_size = 64,
                                            shuffle = True)

        testset = torchvision.datasets.MNIST(root="./data/",
                        transform = transform,
                        train = False)

        testloader = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size = 64,
                                            shuffle = False)
        return args, trainloader, testloader
    else:
        print ('Data load error!')
        return 0

def Set_model(net, client, args):
    print('==> Building model..')
    Model = [None for i in range (client)]
    Optimizer = [None for i in range (client)]
    if net == 'MNISTNet':
        for i in range (client):
            Model[i] = MNISTNet()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        global_model = MNISTNet()
        return Model, global_model, Optimizer
    elif net == 'MobileNet':
        for i in range (client):
            Model[i] = MobileNet()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = MobileNet()
        return Model, global_model, Optimizer
    elif net == 'ResNet18':
        for i in range (client):
            Model[i] = ResNet18()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = ResNet18()
        return Model, global_model, Optimizer

def Train(model, optimizer, client, trainloader):
    criterion = nn.CrossEntropyLoss().to(device)
    # cpu ? gpu
    for i in range(client):
        model[i] = model[i].to(device)
    P = [None for i in range (client)]

    # share a common dataset
    train_loss = [0 for i in range (client)]
    correct = [0 for i in range (client)]
    total = [0 for i in range (client)]
    Loss = [0 for i in range (client)]
    time_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx < 360:
                idx = (batch_idx % client)
                model[idx].train()
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer[idx].zero_grad()
                outputs = model[idx](inputs)
                Loss[idx] = criterion(outputs, targets)
                Loss[idx].backward()
                optimizer[idx].step()
                train_loss[idx] += Loss[idx].item()
                _, predicted = outputs.max(1)
                total[idx] += targets.size(0)
                correct[idx] += predicted.eq(targets).sum().item()
    time_end = time.time()
    if device == 'cuda':
        for i in range (client):
            model[i].cpu()
    for i in range (client):
        P[i] = copy.deepcopy(model[i].state_dict())

    return P, (time_end-time_start)

def Test(model, testloader):
    # cpu ? gpu
    model = model.to(device)

    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        indx_target = target.clone()
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
    test_loss = test_loss / len(testloader) # average over number of mini-batch
    accuracy = float(correct / len(testloader.dataset))
    if device == 'cuda':
        model.cpu()
    return accuracy, test_loss

def Aggregate(model, client):
    # P = []
    # for i in range (client):
    #     P.append(copy.deepcopy(model[i].state_dict()))
    # Q = model[i].state_dict()
    # for key in P[0].keys():
    #     for i in range (client):
    #         if i != 0:
    #             P[0][key] += P[i][key]
    #     P[0][key] = torch.true_divide(P[0][key],client)

    P = copy.deepcopy(model[0].state_dict())
    Q = copy.deepcopy(model[1].state_dict())
    for key in P.keys():
        P[key] = P[key] + P[key]
        P[key] = torch.true_divide(P[key],2)
    for key in P[0].keys():
        if key == 'layers.1.bn1.weight':
            print(P[key])
    return P


def run(dataset, net, client):
    X, Y, Z = [], [], []
    args, trainloader, testloader = Set_dataset(dataset)
    model, global_model, optimizer = Set_model(net, client, args)
    pbar = tqdm(range(args.epoch))
    start_time = 0
    for i in range (args.epoch):
        Temp, process_time = Train(copy.deepcopy(model), optimizer, client, trainloader)
        for j in range (client):
            model[j].load_state_dict(Temp[j])
        global_temp = MobileNet()
        global_temp.load_state_dict(Aggregate(copy.deepcopy(model), client))
        acc, loss = Test(global_temp, testloader)
        for j in range (client):
            model[j].load_state_dict(global_temp.state_dict())
        start_time += process_time
        pbar.set_description("Epoch: %d Accuracy: %.3f Loss: %.3f Time: %.3f" %(i, acc, loss, start_time))
        X.append(start_time)
        Y.append(acc)
        Z.append(loss)
    location = '/home/cifar-gcn-drl/Test_data/FedAVG.csv'
    dataframe = pd.DataFrame(times, columns=['X'])
    dataframe = pd.concat([dataframe, pd.DataFrame(score,columns=['Y'])],axis=1)
    dataframe = pd.concat([dataframe, pd.DataFrame(loss,columns=['Z'])],axis=1)
    dataframe.to_csv(location,mode = 'w', header = False,index=False,sep=',')

if __name__ == '__main__':
    run(dataset = 'CIFAR10', net = 'MobileNet', client = 10)
