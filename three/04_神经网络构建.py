"""
nn神经网络,构建神经网络就是在构建神经元
神经元由加权求和+激活函数构成
神经网络由输入层、隐藏层和输出层、
相邻层神经元相互连接，同层神经元相互隔离
有几个特征就有几个神经元
前层输出=下层输入
Sigmoid激活函数映射成概率
激活函数对每层的数据植入非线性因素，提升网络对复杂问题的你拟合问题
没有激活函数代表是一个大的线性模型

1)Sigmoid激活函数
输出层（浅层）
5层之内就会产生梯度消失,适用于二分类
f(x)=1/1+e(-x),x越大,整体越大。 会把任意的输入映射到(0,1)之间
[-6,6]之间有效果 [-3,3]之间效果明显   求导大概以0.5为中心，[0,0.25]

2)tanh激活函数
可在隐藏层（浅层）用,Sig在输出层使用
梯度比Sig大,减少迭代次数
f(x)=1-e(-2x)/1+e(-2x),x越大,整体越接近1,x越小,整体越接近-1
会把任意的输入映射到(-1 ,1)之间
[-3,3]之间有效果 [-1,1]之间效果明显   求导大概以0为中心，[0,1]

3)ReLU激活函数
缓解过拟合 速度更快效率更高 适合深层神经网络
f(x)=max(0,x) 默认情况只考虑正样本,将小于0的值映射为0,大于0的保持不变
导数=0或者1。x<0时导数为0 x>0时导数为1，能保持梯度不衰减，缓解梯度消失问题
  
隐藏层：
优先使用ReLU 效果不好尝试Leak
少使用Sig,可尝试tanh

输出层：
二分类选择sig,多分类选择softmax,回归问题选择identify(f(x)=x)
"""
import torch
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False


#Sig,tanh,ReLU函数图像
fig,axes = plt.subplots(1,2)
x = torch.linspace(-20,20,1000)
y = torch.relu(x)
axes[0].plot(x,y)
axes[0].set_title("relu激活函数图像")
axes[0].grid()
x = torch.linspace(-20,20,1000,requires_grad=True)

torch.relu(x).sum().backward()

axes[1].plot(x.detach(),x.grad)
axes[1].set_title("relu激活导数函数图像")
axes[1].grid()
plt.show()
