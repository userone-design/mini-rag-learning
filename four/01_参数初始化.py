"""
参数主要有权重和偏置(一般为0)
参数目的:
1.防止梯度消失和梯度爆炸
2.提高模型的收敛速度
3.为了打破对称性，使得每个神经元都能学习到不同的特征

nn.init.uniform() 随机均匀分布(0,1)
nn.init.normal() 正态分布 均值为0,标准差为1
nn.init.zeros_() 全0初始化
nn.init.ones() 全1 √常用
nn.init.constant_() 固定值分布

fan_in:输入神经元的个数 当前层=上一层输出

kaming/HE初始化 
分为正态分布的kaming_nuiform()  均匀分布的kaming_uniform()
xavier初始化


演示参数初始化的七种方式
参数初始化的方式：
1.无法打破对称性：
  全0初始化,全1初始化,固定值

2.打破对称性：
  随机初始化
  正态分布初始化
  Kaiming初始化
  Xavier初始化

  
1.记忆kaming初始化和xavier初始化,全0初始化
2.关于初始化的选择
  ReLU/Leaky ReLU推荐使用Kaiming初始化
  Sigmoid/Tanh推荐使用Xavier初始化

  浅层网络,推荐使用随机初始化
  深层网络,推荐使用Kaiming初始化和Xavier初始化
"""

import torch.nn as nn
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

#1.均匀随机分布初始化
def dm01():
    #创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    #对权重进行随机初始化
    nn.init.uniform_(linear.weight)
    #对偏置进行随机初始化
    nn.init.uniform_(linear.bias)
    print("权重初始化为：", linear.weight)
    print("偏置初始化为：", linear.bias)


#2.固定值初始化
def dm02():
    #创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    #对权重进行固定值初始化
    nn.init.constant_(linear.weight,3 )
    #对偏置进行固定值初始化
    nn.init.constant_(linear.bias, 3)
    print("权重初始化为：", linear.weight.data)
    print("偏置初始化为：", linear.bias.data)

#3.全0初始化
def dm03():
    #创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    #对权重进行全0初始化，全0 
    nn.init.zeros_(linear.weight)
    #对偏置进行全0初始化
    nn.init.zeros_(linear.bias)
    print("权重初始化为：", linear.weight.data)
    print("偏置初始化为：", linear.bias.data)


#4.全1初始化
def dm04():
    #创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    #对权重进行全1初始化
    nn.init.ones_(linear.weight)
    #对偏置进行全1初始化
    nn.init.ones_(linear.bias)
    print("权重初始化为：", linear.weight.data)
    print("偏置初始化为：", linear.bias.data)


#5.正太分布初始化
def dm05():
    #创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    #对权重进行正太分布初始化
    nn.init.normal_(linear.weight, mean=0, std=1)
    #对偏置进行正太分布初始化
    nn.init.normal_(linear.bias, mean=0, std=1)
    print("权重初始化为：", linear.weight.data)
    print("偏置初始化为：", linear.bias.data)


#6.kaiming初始化
def dm06():
    #创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    #对权重进行kaiming初始化
    nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
    #对偏置进行kaiming初始化
    nn.init.kaiming_normal_(linear.bias, mode='fan_in', nonlinearity='relu')
    #nn.init.kaiming_normal_(linear.weight, mode='fan_out', nonlinearity='relu')
    print("权重初始化为：", linear.weight.data)
    print("偏置初始化为：", linear.bias.data)

#7.xavier初始化
def dm07():
    #创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    #对权重进行xavier初始化
    nn.init.xavier_normal_(linear.weight)
    #对偏置进行xavier初始化
    nn.init.xavier_normal_(linear.bias)
    #nn.init.xavier_uniform_(linear.weight)
    #nn.init.xavier_uniform_(linear.bias)
    print("权重初始化为：", linear.weight.data)
    print("偏置初始化为：", linear.bias.data)

















if __name__ == '__main__':
   # dm01()
   # dm02()
    #dm03()
    #dm04()
    #dm05()
    #dm06()
    dm07()












