import pandas as pd
import numpy as np
import tensorflow as tf



PATH = "./ssq.txt"
data = open(PATH)
lines = data.readlines()
ab = np.array(range(1,34))
ab = ab[:,np.newaxis]
a = np.zeros((33,6))                #生成矩阵
b = np.zeros((33,1))
#a = [[0]*6 for i in range(33)]
data_num = []

#print(type(a))

for i in range(len(lines)-1,-1,-1):
    datetime = lines[i].split(",")[0]
    id = lines[i].split(",")[1]
    num_last = int(lines[i].split(",")[3])
    for k in range(1,17):
        if k ==num_last:
            b[k-1][0] +=1
    num = lines[i].split(",")[2]

    num1 = int(num.split(" ")[1])
    num2 = int(num.split(" ")[2])
    num3 = int(num.split(" ")[3])
    num4 = int(num.split(" ")[4])
    num5 = int(num.split(" ")[5])
    num6 = int(num.split(" ")[6])
    #print(num1,num2,num3,num4,num5,num6)
    data_num.append((datetime,id,num1,num2,num3,num4,num5,num6,num_last))
    for j in range(1,34):
        if j == num1:
            a[j-1][0]+=1
        elif j == num2:
            a[j-1][1]+=1
        elif j == num3:
            a[j-1][2]+=1
        elif j == num4:
            a[j-1][3]+=1
        elif j == num5:
            a[j-1][4]+=1
        elif j == num6:
            a[j-1][5]+=1

all = np.concatenate((ab,a,b),axis=1)
data = pd.DataFrame(all,columns=["数字","红1","红2","红3","红4","红5","红6","蓝"],dtype=np.dtype('int64'))
data1 = pd.DataFrame(data_num,columns=["时间","期数","红1","红2","红3","红4","红5","红6","蓝"],dtype=np.dtype('int64'))
data.to_csv("./data_show.csv")
data1.to_csv("./data.csv")
print(data.to_string(index=False))



























