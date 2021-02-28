import matplotlib.pyplot as plt
import numpy as np

# accuracy
x1, y1 = np.loadtxt('/home/mnist-gcn-drl/Test_data/FedAVG_mnist_ACC.csv',delimiter=',',unpack = True)
plt.plot(x1,y1, color = [0.1,0.53,0.93], label = 'FedAVG')
x11,y11 = np.loadtxt('/home/mnist-gcn-drl/Test_data/mnist_acc_1.csv',delimiter=',',unpack = True)
plt.plot(x11,y11, color = [0.75,0.24,1], label = 'DRL')


plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('MNIST-Comparasion')
plt.legend()
fig = plt.gcf()
fig.savefig("/home/mnist-gcn-drl/Fig/MNIST_Accuracy.eps")
plt.cla()


# loss
x2, y2 = np.loadtxt('/home/mnist-gcn-drl/Test_data/FedAVG_mnist_LOSS.csv',delimiter=',',unpack = True)
plt.plot(x2,y2, color = [0.1,0.53,0.93], label = 'FedAVG')
x21,y21 = np.loadtxt('/home/mnist-gcn-drl/Test_data/mnist_loss_1.csv',delimiter=',',unpack = True)
plt.plot(x21,y21, color = [0.75,0.24,1], label = 'DRL')

plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('MNIST-Comparasion')
plt.legend()
fig1 = plt.gcf()
fig1.savefig("/home/mnist-gcn-drl/Fig/MNIST_Loss.eps")
plt.cla()


x3, y3 = np.loadtxt('/home/cifar-gcn-drl/Test_data/FedAVG_cifar10_ACC.csv',delimiter=',',unpack = True)
plt.plot(x3,y3, color = [0.1,0.53,0.93], label = 'FedAVG')
x31, y31 = np.loadtxt('/home/cifar-gcn-drl/Test_data/FedAVG_multi_cifar10_ACC.csv',delimiter=',',unpack = True)
plt.plot(x31,y31, color = [0.75,0.24,1], label = 'Multi')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Comparasion')
plt.legend()
fig1 = plt.gcf()
fig1.savefig("/home/cifar-gcn-drl/Fig/CIFAR10_Multi_comparasion.eps")
plt.cla()


