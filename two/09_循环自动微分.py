"""

"""
import torch
w = torch.tensor(10,requires_grad= True,dtype=torch.float32)

loss = w**2+20

print(f"初始:{w},(0.01*w.grad):无,loss:{loss}")
#正向传播
for i in range(1,101):
    loss = w**2 + 20
    if w.grad is not None:
        w.grad.zero_()
    loss.sum().backward()
    w.data = w.data-0.01*w.grad
    print(f"第{i}次，权重：{w},(0.01*w.grad):{0.01*w.grad},loss:{loss}")

print(f"权重：{w},梯度值：{w.grad},loss:{loss}")









