import matplotlib.pyplot as plt
import numpy as np
import csv

# accuracy
x1, y1 = np.loadtxt('/home/cifar-gcn-drl/Test_data/FedAVG_ACC.csv',delimiter=',',unpack = True)


plt.plot(x1,y1, [0.1,0.53,0.93], label = 'FedAVG')
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('Comparasion')
plt.legend()

plt.savefig("/home/cifar-gcn-drl/Fig/Accuracy.eps")
plt.show()


# loss
x2, y2 = np.loadtxt('/home/cifar-gcn-drl/Test_data/FedAVG_LOSS.csv',delimiter=',',unpack = True)


plt.plot(x2,y2, [0.75,0.24,1], label = 'FedAVG')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Comparasion')
plt.legend()

plt.savefig("/home/cifar-gcn-drl/Fig/Loss.eps")