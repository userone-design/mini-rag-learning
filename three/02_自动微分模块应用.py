"""
先前向传播计算出z
就i损失函数结合z和y，计算梯度
结合权重更新公式，更新权重
"""

import torch
#定义x，表示特征
x = torch.ones(2,5)
print(f"x:{x}")
#定义y表示真实值
y = torch.zeros(2,3)
print(f"y:{y}")

w = torch.randn(5,3,requires_grad=True)#x @ w+b x的列=w的行，结果（y）的列是w的列
print(f"w:{w}")

b = torch.randn(3,requires_grad=True)
print(f"b:{b}")
#计算z预测值
z = torch.matmul(x,w) + b
#z = x @ w + b
print(f"z:{z}")
#定义损失函数
criterion = torch.nn.MSELoss() 
loss = criterion(z,y) 
print(f"loss:{loss}")
#自动微分 求导 结合反向传播，更新权重
loss.backward()
print(f"w的梯度：{w.grad}")
print(f"b的梯度：{b.grad}")













