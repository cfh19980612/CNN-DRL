import os
# os.environ['CUDA_ENABLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import random
import argparse
import copy
import pandas as pd
import numpy as np
from utils import progress_bar
from models import *
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

    # CNN training process
#     def CNN_train(self, i, criterion):
#         self.Model[i] = self.Model[i].to(self.device)
#         # gpu ?
#         if self.device == 'cuda':
#             self.Model[i] = torch.nn.DataParallel(self.Model[i])
#             cudnn.benchmark = True
#         self.Model[i].train()

#         # training
#         train_loss = 0
#         correct = 0
#         total = 0
#         Loss = 0
#         for batch_idx, (inputs, targets) in enumerate(self.trainloader):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             self.Optimizer[i].zero_grad()
#             outputs = self.Model[i](inputs)
#             Loss = criterion(outputs, targets)
#             Loss.backward()
#             self.Optimizer[i].step()

#             train_loss += Loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#         if self.device == 'cuda':
#             self.Model[i].cpu()


    # multiple processes to train CNN models
    def CNN_processes(self, Model, Optimization, client, trainloader):
        # loss func
        criterion = nn.CrossEntropyLoss()

        # cpu ? gpu
        for i in range(Client):
            Model[i] = self.Model[i].to(self.device)
        P = [None for i in range (Client)]

#         # each silo owns a complete dataset
#         for client in range (Client):
#             self.Model[client].train()
#             train_loss = 0
#             correct = 0
#             total = 0
#             Loss = 0
#             for batch_idx, (inputs, targets) in enumerate(self.trainloader):
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 self.Optimizer[client].zero_grad()
#                 outputs = self.Model[client](inputs)
#                 Loss = criterion(outputs, targets)
#                 Loss.backward()
#                 self.Optimizer[client].step()

#                 train_loss += Loss.item()
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()
#                     progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                                 % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # share a common dataset
        train_loss = [0 for i in range (Client)]
        correct = [0 for i in range (Client)]
        total = [0 for i in range (Client)]
        Loss = [0 for i in range (Client)]
        for batch_idx, (inputs, targets) in enumerate(trainloader):
                if batch_idx < 10:
                    client = (batch_idx % Client)
                    Model[client].train()
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    Optimizer[client].zero_grad()
                    outputs = Model[client](inputs)
                    Loss[client] = criterion(outputs, targets)
                    Loss[client].backward()
                    Optimizer[client].step()

                    train_loss[client] += Loss[client].item()
                    _, predicted = outputs.max(1)
                    total[client] += targets.size(0)
                    correct[client] += predicted.eq(targets).sum().item()

#                     progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                                 % (train_loss[client]/(batch_idx+1), 100.*correct[client]/total[client], correct[client], total[client]))

        if self.device == 'cuda':
            for i in range (Client):
                Model[i].cpu()
        for i in range (Client):
            P[i] = copy.deepcopy(Model[i].state_dict())

        return P

    # CNN_test
    def CNN_test(self, model, testloader):
        # cpu ? gpu
        model = model.to(self.device)

        model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.testloader:
            indx_target = target.clone()
            if self.device == 'cuda':
                data, target = data.cuda(), target.cuda()
#             with torch.no_grad(data,target):

            output = model(data)
            test_loss += F.cross_entropy(output, target).data
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.cpu().eq(indx_target).sum()

        test_loss = test_loss / len(testloader) # average over number of mini-batch
        accuracy = float(correct / len(testloader.dataset))
        if self.device == 'cuda':
            model.cpu()
        return accuracy

    # local_aggregate
    def Local_agg(self, Model, model, i, Client, Imp, latency):
        # print ('Action: ',p)
        Imp = np.array(Imp).reshape((Client,Client))
        # print ('P: ', p)
        time = 0
        Q = []
        P = copy.deepcopy(model.state_dict())
        for j in range (Client):
            Q.append(copy.deepcopy(Model[j].state_dict()))
        for key, value in P.items():
            m = 0
            for j in range (Client):
                if i != j:
                    if Imp[i,j] > 0:
#                     P[key] = P[key] + (Imp[i,j]/Imp[i].sum())*Q[j][key]
                        P[key] = P[key] + Q[j][key]
                        m += 1
            P[key] = torch.true_divide(P[key],m+1)
            # P[key] = P[key]/m+1

        for j in range (Client):
            # if self.G.has_edge(i,j):
            time += latency[i][j]
        return P, time

    # Global aggregate
    def Global_agg(self, Client):

        P = copy.deepcopy(self.Model[0].state_dict())
        for key, value in P.items():
            for i in range (1,Client,1):
                temp = copy.deepcopy(self.Model[i].state_dict())
                P[key] = P[key] + temp[key]
            P[key] = torch.true_divide(P[key],Client)
        return temp

    # step time cost
    def step_time(self, T):
        time = max(T)
        return time

    # to CSV
    def toCsv(self, times, score):
        dataframe = pd.DataFrame(times, columns=['X'])
        dataframe = pd.concat([dataframe, pd.DataFrame(score,columns=['Y'])],axis=1)
        dataframe.to_csv('/home/CIFAR10/Test_data/test.csv',mode = 'w', header = False,index=False,sep=',')

    # return model
    def toModel(self):
        return self.Model


    # def forward(self, epoches, Client):
    #     times, score = [], []
    #     t = 0
    #     args, trainloader, testloader = self.Set_dataset()
    #     self.Set_Environment(args)

    #     global_model = MobileNet() if self.net == 'MobileNet' else VGG('VGG19')

    #     # GAT network
    #     net = GATLayer(self.g,in_dim = 864,out_dim = 20)

    #     for epoch in range(0, epoches):

    #         Tim, Loss = [], []
    #         # Loss = [0 for i in range (Client)]

    #         P = self.CNN_train(epoch,trainloader)

    #         for i in range (Client):
    #             self.Model[i].load_state_dict(P[i])

    #         # global model
    #         # global_model = self.Global_agg()

    #         accuracy = self.CNN_test(epoch,self.Model[0],testloader)

    #         score.append(accuracy)

    #         # aggregate local model
    #         # Step 1: calculate the weight for each neighborhood
    #         net.update_graph(self.Model, Client)
    #         net.forward()
    #         # Step 2: aggregate the model from neighborhood
    #         for i in range (5):
    #             P_new = [None for m in range (Client)]
    #             for x in range (Client):
    #                 P_new[x], temp = self.Local_agg(self.Model[x],x)
    #                 Tim.append(temp)
    #         # update
    #         for client in range (Client):
    #             self.Model[client].load_state_dict(P_new[client])

    #         times, t = self.step_time(times, Tim, t)


    #         self.toCsv(times,score)