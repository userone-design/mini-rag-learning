"""
使用sigmoid函数作为激活函数的二分类任务损失函数
使用nn.BCELoss()，该函数内部会自动调用sigmoid函数计算预测概率
并计算二分类交叉熵损失   nn.BCELoss() = sigmoid+损失计算
二分类交叉熵损失函数公式：
L = -Σ(yi * log(ŷi) + (1-yi) * log(1-ŷi))        ŷi=加权求和+激活函数=sigmoid(zi)
其中，yi是真实标签，ŷi是预测概率，L用来衡量模型


"""

import torch
import torch.nn as nn
#1.定义函数
def dm01():
    #1.创建真实值
    y_true = torch.tensor([0, 1, 0],dtype=torch.float32) 
    #2.创建预测值
    y_pred = torch.tensor([0.69, 0.54, 0.26],requires_grad=True,dtype=torch.float32)
    #3.创建二分类交叉熵损失函数
    criterion = nn.BCELoss()
    #4.计算损失
    loss = criterion(y_pred, y_true).detach().numpy() #detach()将计算图分离，防止梯度回传
    print(f"真实值：{y_true},预测值：{y_pred},损失值：{loss}")


if __name__ == "__main__":
    dm01()