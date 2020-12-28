import gym
from CNN import cnn
import random
import math
import networkx as nx
import numpy as np

class FedEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    def __init__(self, Client, k ):
        
        self.client = Client
        self.p = 0.5
        self.Model = []

        # small world
        self.G = nx.watts_strogatz_graph(n = self.client, k = k, p = self.p)

        # To DGL graph
        # self.g = dgl.from_networkx(self.G)

        # latency simulation
        self.latency = [[0 for i in range (self.client)]for j in range (self.client)]
        for i in range (self.client):
            for j in range (self.client):
                self.latency[i][j] = random.randint(1,20)

        self.task = cnn(Client = self.client, Dataset = 'CIFAR10', Net = 'MobileNet')    # num of clients, num of neighbors, dataset, network
        self.Model, self.global_model = self.task.Set_Environment(Client)

    

    def step(self, action, epoch):

        # # GAT network
        # net = GATLayer(self.g,in_dim = 864,out_dim = 20)

        Tim, accuracy_list = [], []
        # Loss = [0 for i in range (Client)]

        P = self.task.CNN_train(epoch, self.client)

        for i in range (self.client):
            self.Model[i].load_state_dict(P[i])

        # global model   
        # self.global_model.load_state_dict(self.task.Global_agg(self.client)) 
        
        accuracy = self.task.CNN_test(epoch,self.Model[0])

        # aggregate local model
        # Step 1: calculate the weight for each neighborhood
        # Step 2: aggregate the model from neighborhood
        for i in range (1):
            P_new = [None for m in range (self.client)]
            for x in range (self.client):
                P_new[x],temp = self.task.Local_agg(self.Model[x],x,self.client,action,self.latency)
                
                Tim.append(temp)
        # update     
        for client in range (self.client):
            self.Model[client].load_state_dict(P_new[client])

        t = self.task.step_time(Tim)


        # PCA
        parm_local = {}
        Name = []
        S_local = [None for i in range (self.client)]
        for i in range (self.client):
            S_local[i] = []
            for name, parameters in self.Model[i].named_parameters():
                # print(name,':',parameters.size())
                parm_local[name]=parameters.detach().cpu().numpy()
                Name.append(name)
            
            for a in parm_local[Name[0]][0::].flatten():
                    aa = a
                    S_local[i].append(aa)
        s = np.array(S_local).flatten()
        
        # self.toCsv(times,score)
        reward = pow(32, accuracy-0.8)-1

        return t, accuracy, s, reward



    
    def reset(self):
        self.Model, global_model = self.task.Set_Environment(self.client)
        # PCA
        parm_local = {}
        Name = []
        S_local = [None for i in range (self.client)]
        for i in range (self.client):
            S_local[i] = []
            for name, parameters in self.Model[i].named_parameters():
                # print(name,':',parameters.size())
                parm_local[name] = parameters.detach().cpu().numpy()
                Name.append(name)
            
            for a in parm_local[Name[0]][0::].flatten():
                    aa = a
                    S_local[i].append(aa)
        s = np.array(S_local).flatten()
        print (s.size)

        return s
    
    def save_acc(self, X, Y):
        self.task.toCsv(X,Y)

    def render(self, mode='human'):
        return None
        
    def close(self):
        return None
