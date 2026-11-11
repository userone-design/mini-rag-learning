"""
损失函数：用于衡量模型预测值与真实值之间的差异，衡量模型的质量

分类：
   分类问题：
   多分类交叉熵损失：CrossEntropyLoss
   二分类交叉熵损失：BCELoss

   回归问题：
   MAE损失函数：Mean Absolute Error,平均绝对误差
   MSE损失函数：Mean Squared Error，均方误差
   Smooth L1损失函数：Smooth L1 Loss
 
多分类交叉熵损失CrossEntropyLoss = softmax+损失计算
输出层不用额外加softmax函数，nn.CrossEntropyLoss()内部会自动调用softmax函数计算预测概率，并计算交叉熵损失
多分类交叉熵损失函数公式：
L = -Σ(yi * log(ŷi))        ŷi=加权求和+激活函数=softmax(zi) 
其中，yi是真实标签，ŷi是预测概率，L用来衡量模型预测值与真实值之间的差异

目的：最小化正确类别的预测概率最大化，最小化错误类别的预测概率
多分类任务损失函数使用nn.CrossEntropyLoss()，该函数内部会自动调用softmax函数计算预测概率，并计算交叉熵损失
""" 
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

#1.定义函数
def dm01():
    #1.创建真实值
   # y_true = torch.tensor([[0, 1, 0],[1,0,0]]) 
    y_true = torch.tensor([1,2])
    #2.创建预测值
    y_pred = torch.tensor([[0.1, 0.8, 0.1],[0.7, 0.2, 0.1]],requires_grad=True,dtype=torch.float32)
    #3.创建损失函数
    criterion = nn.CrossEntropyLoss()
    #4.计算损失
    loss = criterion(y_pred, y_true)
    print(f"真实值：{y_true},预测值：{y_pred},损失值：{loss}")


if __name__ == "__main__":
    dm01()
