import gym
from CNN import cnn
import random
import math
import networkx as nx
import numpy as np
import argparse
from sklearn.decomposition import PCA
from models import *

import torch
import torchvision
import torchvision.transforms as transforms

class FedEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    def __init__(self, Client, k, dataset, net):

        self.client = Client
        self.p = 0.5
        self.dataset = dataset
        self.net = net
        # small world
        self.G = nx.watts_strogatz_graph(n = self.client, k = k, p = self.p)

        # To DGL graph
        # self.g = dgl.from_networkx(self.G)

        # PCA
        self.pca = PCA(n_components = self.client)
        # latency simulation
        self.latency = [[0 for i in range (self.client)]for j in range (self.client)]
        for i in range (self.client):
            for j in range (self.client):
                self.latency[i][j] = random.randint(1,20)

        self.task = cnn()    # num of clients, num of neighbors, dataset, network
        self.args, self.trainloader, self.testloader = self.Set_dataset()
        self.Model, self.global_model, self.Optimizer = self.Set_Environment(Client)

    # Preparing data
    def Set_dataset(self):
        if self.dataset == 'CIFAR10':
            parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
            parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
            parser.add_argument('--resume', '-r', action='store_true',
                                help='resume from checkpoint')
            args = parser.parse_args()
            best_acc = 0  # best test accuracy
            start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
        elif self.dataset == 'MNIST':
            parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
            parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
            parser.add_argument('--resume', '-r', action='store_true',
                                help='resume from checkpoint')
            args = parser.parse_args()
            best_acc = 0  # best test accuracy
            start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
    # building models
    def Set_Environment(self, Client):
        print('==> Building model..')
        Model = [None for i in range (Client)]
        Optimizer = [None for i in range (Client)]
        if self.dataset == 'MNIST':
            for i in range (Client):
                Model[i] = MNISTNet()
                Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=self.args.lr,
                                momentum=0.9, weight_decay=5e-4)
            global_model = MNISTNet()
            return Model, global_model, Optimizer
        elif self.dataset == 'CIFAR10':
            for i in range (Client):
                Model[i] = MobileNet()
                Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=self.args.lr,
                            momentum=0.9, weight_decay=5e-4)
            global_model = MobileNet()
            return Model, global_model, Optimizer



    def step(self, action, epoch):

        # # GAT network
        # net = GATLayer(self.g,in_dim = 864,out_dim = 20)

        Tim, accuracy_list = [], []
        # Loss = [0 for i in range (Client)]

        P = self.task.CNN_processes(self.Model, self.Optimizer, self.client, self.trainloader)

        for i in range (self.client):
            self.Model[i].load_state_dict(P[i])

        # global model
        # self.global_model.load_state_dict(self.task.Global_agg(self.client))

        accuracy = self.task.CNN_test(self.Model[0],self.testloader)

        # aggregate local model
        # Step 1: calculate the weight for each neighborhood
        # Step 2: aggregate the model from neighborhood
        for i in range (1):
            P_new = [None for m in range (self.client)]
            for x in range (self.client):
                P_new[x],temp = self.task.Local_agg(self.Model[x+1],x,self.client,action,self.latency)

                Tim.append(temp)
        # update
        for client in range (self.client):
            self.Model[client].load_state_dict(P_new[client])

        t = self.task.step_time(Tim)


        # PCA
        parm_local = {}
        S_local = [None for i in range (self.client)]

        for i in range (self.client):
            S_local[i] = []
            Name = []
            for name, parameters in self.Model[i+1].named_parameters():
                # print(name,':',parameters.size())
                parm_local[name]=parameters.detach().cpu().numpy()
                Name.append(name)
            for j in range(len(Name)):
                for a in parm_local[Name[j]][0::].flatten():
                    S_local[i].append(a)
            S_local[i] = np.array(S_local[i]).flatten()
        # to 1-axis

        # convert to [num_samples, num_features]
        S = np.reshape(S_local,(self.client,3217226))

        # pca
        state = self.pca.fit_transform(S)
        state = state.flatten()
        # self.toCsv(times,score)
        reward = pow(64, accuracy-0.85)-1
        return t, accuracy, state, reward



    def reset(self, Tag):
        # self.Model = self.task.returnModel(self.client)
        # PCA
        parm_local = {}
        S_local = [None for i in range (self.client)]
        for i in range (self.client):
            S_local[i] = []
            Name = []
            for name, parameters in self.Model[i].named_parameters():
                # print(name,':',parameters.size())
                parm_local[name]=parameters.detach().cpu().numpy()
                Name.append(name)
            for j in range(len(Name)):
                for a in parm_local[Name[j]][0::].flatten():
                    S_local[i].append(a)
            S_local[i] = np.array(S_local[i]).flatten()
        # to 1-axis
        S_local = np.array(S_local).flatten()

        # convert to [num_samples, num_features]
        S = np.reshape(S_local,(self.client,3217226))

        # pca training ?
        if Tag:
            self.pca.fit(S)
        state = self.pca.fit_transform(S)
        state = state.flatten()
#             print('without flatten: ',S_local[i].shape)
#             S_local[i] = S_local[i].flatten().reshape(1,-1)
#             print('without pca: ',S_local[i].shape)
#             S_local[i] = pca.fit(S_local[i])
#             print('with pca: ',S_local[i].shape)
#         s = np.array(S_local).flatten()


        return state
    
    def save_acc(self, X, Y):
        self.task.toCsv(X,Y)

    def render(self, mode='human'):
        return None
        
    def close(self):
        return None
