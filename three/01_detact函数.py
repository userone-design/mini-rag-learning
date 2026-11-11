"""
如果设置了自动微分，就无法直接转numpy
"""
import numpy as np
import torch
t1 = torch.tensor([10,20],requires_grad=True,dtype=float)

# n1 = t1.numpy()
# print(f"{n1},{type(n1)}")
n1 =  t1.detach().numpy()
print(f"{n1},{type(n1)}")

print(f"t1:{t1.requires_grad}")
print(f"n1:{n1.requires_grad}")



