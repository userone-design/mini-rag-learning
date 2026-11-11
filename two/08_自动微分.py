"""
自动微分模块 结合 反向传播，更新W,b
只支持标量张量对向量张量的求导
backward()反向传播
forward()正向传播

权重更新：W新=W旧-学习率*梯度（损失函数的导数）

"""
import torch
#记录初始权值,require ：是否自动微分
w = torch.tensor(10,requires_grad=True,dtype=torch.float)
#定义loss
loss = 2*w**2
print({type(loss.grad_fn)})
print(loss.sum())

#计算梯度
loss.sum().backward() #会自动返回到grad
#loss.backward()

#带入权重更新公式
w.data = w.data-0.01*w.grad

print(f"更新后:{w}")
 



