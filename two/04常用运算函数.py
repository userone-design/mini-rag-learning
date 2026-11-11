"""
sum(),max(),min(),mean()   dim:0表示列，1表示行

pow() sqrt() exp() log() log2() log10()
要求：
sum()z,max(),min(),mean() pow()n次幂 **

mean 都要用float

"""
import torch
t1 = torch.tensor([[1,2,3],[4,5,6]],dtype=float)

print(t1.sum(dim=0))
print(t1.sum(dim=1))
print(t1.sum())
print("-"*30)

print(t1.max(dim=0))
print(t1.max(dim=1))
print(t1.max())
print("-"*30)


print(t1.mean(dim=0))
print(t1.mean(dim=1))
print(t1.mean())
print("-"*30)

print(t1.pow(2))
print(t1.pow(3))
print(t1**3)
print("-"*30)

print(t1.sqrt())
print("-"*30)

print(t1.exp())
print("-"*30)

