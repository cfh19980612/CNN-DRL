import os
# os.environ['CUDA_ENABLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import copy
import pandas as pd
import numpy as np
import time
from utils import progress_bar
from multiprocessing import Pool
import queue

class cnn(nn.Module):
    def __init__(self):
        self.p = 0.5
        # cpu ? gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def returnModel(self, Client):
        P = []
        for i in range (Client):
            P.append(self.Model[i+1])
        return P

    # multiple processes to train CNN models
    def CNN_processes(self, Model, Optimizer, Client, trainloader):
        criterion = nn.CrossEntropyLoss().to(self.device)
        # cpu ? gpu
        for i in range(Client):
            Model[i] = Model[i].to(self.device)
        P = [None for i in range (Client)]

        # share a common dataset
        train_loss = [0 for i in range (Client)]
        correct = [0 for i in range (Client)]
        total = [0 for i in range (Client)]
        Loss = [0 for i in range (Client)]
        time_start = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
                if batch_idx < 360:
                    idx = (batch_idx % Client)
                    Model[idx].train()
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    Optimizer[idx].zero_grad()
                    outputs = Model[idx](inputs)
                    Loss[idx] = criterion(outputs, targets)
                    Loss[idx].backward()
                    Optimizer[idx].step()
                    train_loss[idx] += Loss[idx].item()
                    _, predicted = outputs.max(1)
                    total[idx] += targets.size(0)
                    correct[idx] += predicted.eq(targets).sum().item()
        time_end = time.time()
        if self.device == 'cuda':
            for i in range (Client):
                Model[i].cpu()
        for i in range (Client):
            P[i] = copy.deepcopy(Model[i].state_dict())

        return P, (time_end-time_start)

    # CNN_test
    def CNN_test(self, model, testloader):
        # cpu ? gpu
        model = model.to(self.device)
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in testloader:
            indx_target = target.clone()
            if self.device == 'cuda':
                data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                output = model(data)
            test_loss += F.cross_entropy(output, target).data
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.cpu().eq(indx_target).sum()

        test_loss = test_loss / len(testloader) # average over number of mini-batch
        accuracy = float(correct / len(testloader.dataset))
        if self.device == 'cuda':
            model.cpu()
        return accuracy, test_loss.item()

    # local_aggregate
    def Local_agg(self, Model, i, Client, Imp, latency):
        # print ('Action: ',p)
        Imp = np.array(Imp).reshape((Client,Client))
        Probability = Imp[i]    # get the probabilities that client i use to choose aggregations
        sorted_nums = sorted(enumerate(Probability), key=lambda x: x[1])    # sort the probabilities
        idx = [i[0] for i in sorted_nums]    # get the idex from the sorted probabilities

        K = 4    # maximum aggregations

        # print ('P: ', p)
        time = 0
        Q = []
        P = copy.deepcopy(Model[i].state_dict())
        for j in range (Client):
            Q.append(copy.deepcopy(Model[j].state_dict()))
        for key, value in P.items():
            m = 0
            for j in range (Client):
                if i != j:
                    P[key] = P[key] + Q[idx[j]][key]
                    m = m+1
                    if m >= K:
                        break
            P[key] = torch.true_divide(P[key],m+1)
        n = 0
        for j in range (Client):
            if j != i:
                time += latency[i][idx[j]]
                n += 1
                if n >= K:
                    break
        return P, time

    # Global aggregate
    def Global_agg(self, Client, Model):
        P = []
        for i in range (Client):
            P.append(copy.deepcopy(Model[i].state_dict()))
        for key in P[0].keys():
            for i in range (Client):
                if i != 0:
                    P[0][key] =torch.add(P[0][key], P[i][key])
            P[0][key] = torch.true_divide(P[0][key],Client)
        return P[0]

    # step time cost
    def step_time(self, T):
        time = max(T)
        return time

    # to CSV
    def toCsv(self, times, score, loss, i_episode, dataset):
        if dataset == 'MNIST':
            location_acc = '/home/mnist-gcn-drl/Test_data/mnist_acc_' + str(i_episode) + '.csv'
            location_loss = '/home/mnist-gcn-drl/Test_data/mnist_loss_' + str(i_episode) + '.csv'
        elif dataset == 'CIFAR10':
            location_acc = '/home/cifar-gcn-drl/Test_data/cifar10_acc_' + str(i_episode) + '.csv'
            location_loss = '/home/cifar-gcn-drl/Test_data/cifar10_loss_' + str(i_episode) + '.csv'
        elif dataset == 'FASHION-MNIST':
            location_acc = '/home/fmnist-gcn-drl/Test_data/fmnist_acc_' + str(i_episode) + '.csv'
            location_loss = '/home/fmnist-gcn-drl/Test_data/fmnist_loss_' + str(i_episode) + '.csv'

        dataframe_1 = pd.DataFrame(times, columns=['X'])
        dataframe_1 = pd.concat([dataframe_1, pd.DataFrame(score,columns=['Y'])],axis=1)
        dataframe_1.to_csv(location_acc,mode = 'w', header = False,index=False,sep=',')

        dataframe = pd.DataFrame(times, columns=['X'])
        dataframe = pd.concat([dataframe, pd.DataFrame(loss,columns=['Y'])],axis=1)
        dataframe.to_csv(location_loss,mode = 'w', header = False,index=False,sep=',')