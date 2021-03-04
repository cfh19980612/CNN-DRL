import matplotlib.pyplot as plt
import numpy as np


# topology
x4, y4 = np.loadtxt('/home/cifar-gcn-drl/Test_data/Asyn_cifar10_ACC_base.csv',delimiter=',',unpack = True)
plt.plot(x4,y4, linestyle = '-.', color = [0.412,0.412,0.412], label = 'Base')
x41,y41 = np.loadtxt('/home/cifar-gcn-drl/Test_data/Asyn_cifar10_ACC_move.csv',delimiter=',',unpack = True)
plt.plot(x41,y41, linestyle = '-', color = [1,0.65,0], label = 'Move')

plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Comparasion')
plt.legend()
fig1 = plt.gcf()
fig1.savefig("/home/cifar-gcn-drl/Fig/CIFAR10_topology.eps")
plt.cla()