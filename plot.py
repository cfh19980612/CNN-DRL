import matplotlib.pyplot as plt
import numpy as np

# accuracy
x1, y1 = np.loadtxt('/home/cifar-gcn-drl/Test_data/FedAVG_ACC.csv',delimiter=',',unpack = True)
plt.plot(x1,y1, [0.1,0.53,0.93], label = 'FedAVG')
x11,y11 = np.loadtxt('/home/cifar-gcn-drl/Test_data/test_acc_1.csv',delimiter=',',unpack = True)
plt.plot(x11,y11, [0.75,0.24,1], label = 'DRL')


plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('Comparasion')
plt.legend()
fig = plt.gcf()
fig.savefig("/home/cifar-gcn-drl/Fig/Accuracy.eps")
plt.cla()


# loss
x2, y2 = np.loadtxt('/home/cifar-gcn-drl/Test_data/FedAVG_LOSS.csv',delimiter=',',unpack = True)


plt.plot(x2,y2, [0.75,0.24,1], label = 'FedAVG')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Comparasion')
plt.legend()
fig1 = plt.gcf()
fig1.savefig("/home/cifar-gcn-drl/Fig/Loss.eps")