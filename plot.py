import matplotlib.pyplot as plt
import numpy as np
import csv

x1, y1 = np.loadtxt('/home/cifar-gcn-drl/Test_data/FedAVG_ACC.csv',delimiter=',',unpack = True)


plt.plot(x1,y1, [0.1,0.53,0.93], label = 'FedAVG')
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('Comparasion')
plt.legend()

plt.savefig("/home/cifar-gcn-drl/Fig/Accuracy.eps")