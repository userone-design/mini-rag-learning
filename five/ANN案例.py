"""
ANN人工神经网络：手机价格分类
背景：
    基于手机的20列特征 预测手机的价格区间 

ANN实例jie
    1.构建数据集
    2.搭建神经网络
    3.模型训练
    4.模型测试


"""
import torch
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch import nn
from torch import optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False


# 1.构建数据集
def create_dataset():
    #1.加载数据集
    data = pd.read_csv('./data/手机价格预测.csv')
    # print(f"data:{data}")
    # print(f"data.shape:{data.shape}")
    #2.获取x特征和y标签
    #:-1不要最后一个 前面全要   ,-1只要最后一个
    x,y = data.iloc[:,:-1],data.iloc[:,-1]
    # print(f"x:{x.head()},{x.shape}")
    # print(f"y:{y.head()},{y.shape}")
    #3.把特征列转化为浮点型
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    #4.数据集划分  
    #参数1：数据集 2：标签 3：训练集占比 4：随机种子 5：样本的分类（参考y的类别进行抽取数据）
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3,stratify=y)
    train_dataset = TensorDataset(torch.tensor(x_train.values),torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values),torch.tensor(y_test.values))
    print(f"train_dataset:{train_dataset} test_dataset:{test_dataset}")

    #                                 #20 充当输入特征数     4 充当输出特征数
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))   #np.unique去重



# 2.搭建神经网络
# 3.模型训练
# 4.模型测试

if __name__ == '__main__':
    train_dataset,test_dataset,input_dim,output_dim = create_dataset()
    print(f"train_dataset:{train_dataset}")
    print(f"test_dataset:{test_dataset}")
    print(f"input_dim:{input_dim}") #20
    print(f"output_dim:{output_dim}") #4