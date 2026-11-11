"""
MAE损失函数：误差绝对值之和的平均数 也称L1loss
   权重可以降为0.5，L1loss在0点梯度不平滑，0点不可导

MSE损失函数：误差平方和的平均数，L2loss 或者欧氏距离 容易关注异常值
常作为正则项  当预测和目标值，会发生梯度爆炸


Smoth L1是光滑后的L1    
"""
import torch
import torch.nn as nn
#1.MSE
def dm01():
    y_true = torch.tensor([1,1,1.9])
    y_pred = torch.tensor([2,2,2],dtype=torch.float32,requires_grad=True)
    cri = nn.MSELoss() #均方误差损失函数
    loss = cri(y_pred, y_true).detach().numpy() #detach()将计算图分离，防止梯度回传
    print(f"真实值：{y_true},预测值：{y_pred},损失值：{loss}")


#1.MAE
def dm02():
    y_true = torch.tensor([1,1,1.9])
    y_pred = torch.tensor([2,2,2],dtype=torch.float32,requires_grad=True)
    cri = nn.L1Loss() #均方误差损失函数
    loss = cri(y_pred, y_true   ).detach().numpy() #detach()将计算图分离，防止梯度回传
    print(f"真实值：{y_true},预测值：{y_pred},损失值：{loss}")


#3.Smoth
def dm03():
    y_true = torch.tensor([1,1,1.9])
    y_pred = torch.tensor([2,2,2],dtype=torch.float32,requires_grad=True)
    cri = nn.SmoothL1Loss() #均方误差损失函数
    loss = cri(y_pred, y_true   ).detach().numpy() #detach()将计算图分离，防止梯度回传
    print(f"真实值：{y_true},预测值：{y_pred},损失值：{loss}")



if __name__ == "__main__":
    dm01()
    dm02()
    dm03()
    