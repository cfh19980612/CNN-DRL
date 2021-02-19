import matplotlib.pyplot as plt
import numpy as np
import csv

File = open('/home/cifar-gcn-drl/Test_data/FedAVG.csv')  # 打开csv文件
Reader = csv.reader(File)  # 读取csv文件
Data = list(Reader)  # csv数据转换为列表
length_vol = len(exampleData)  # 得到数据行数
length_clu = len(exampleData[0])  # 得到每行长度

time = list()
acc = list()
loss = list()

for i in range(length_vol):
    time.append(Data[i][0])
    acc.append(Data[i][1])
    loss.append(Data[i][2])

plt.plot(time, acc, [0.1,0.53,0.93], label = 'FedAVG')
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('Comparasion')
plt.legend()

plt.savefig("Accuracy.eps")