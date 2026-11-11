"""
ones(),zeros(),full()填充指定值
"""
import torch
#全1
t1 = torch.ones(2, 3) # 2行3列
print(f"t1:{t1},type:{type(t1)}")
print("-" * 30)
#t2 三行两列
t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f"t2:{t2},type:{type(t2)}")
print("-" * 30)
#t3 基于t2的形状创建张量，元素全为1
t3 = torch.ones_like(t2) # 生成和t2形状一样的张量，元素全为1
print(f"t3:{t3},type:{type(t3)}")



#全0
t1 = torch.zeros(2, 3) # 2行3列
print(f"t1:{t1},type:{type(t1)}")
print("-" * 30)
#t2 三行两列
t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f"t2:{t2},type:{type(t2)}")
#t3 基于t2的形状创建张量，元素全为1
t3 = torch.zeros_like(t2) # 生成和t2形状一样的张量，元素全为1
print(f"t3:{t3},type:{type(t3)}")


#创建全为 (指定值) 的张量
t1 = torch.full((2, 3), 255) # 2行3列，元素全为255
print(f"t1:{t1},type:{type(t1)}")
print("-" * 30)

t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f"t2:{t2},type:{type(t2)}")
print("-" * 30)

t3 = torch.full_like(t2, 255) # 生成和t2形状一样的张量，元素全为9
print(f"t3:{t3},type:{type(t3)}")


