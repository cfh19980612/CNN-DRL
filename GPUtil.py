import GPUtil
import time
import pandas as pd
time_x = []
gpu_y = []

def N_gpu_util_timer():
    for n in range(250):
        GPUs = GPUtil.getGPUs()
        gpu_load = GPUs[0].load
        time_x.append(n)
        gpu_y.append(gpu_load)
        time.sleep(1)
    # print(gpu_y)
    print('N gpu done')

if __name__ == '__main__':
    N_gpu_util_timer()
    location_acc = '/home/cifar-gcn-drl/Test_data/GPU' + '.csv'
    dataframe_1 = pd.DataFrame(time_x, columns=['X'])
    dataframe_1 = pd.concat([dataframe_1, pd.DataFrame(gpu_y,columns=['Y'])],axis=1)
    dataframe_1.to_csv(location_acc,mode = 'w', header = False,index=False,sep=',')