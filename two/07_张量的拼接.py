"""
cat() 指定的维度拼接，不改变维度数,除了拼接的那个维度其他维度保持一样
stack() 会改变维度数，所有输入张量的形状要相同
"""
import torch
torch.manual_seed(24)
t1 = torch.randint(1,10,(2,3))
print(f"t1:{t1},shape:{t1.shape}")
t2 = torch.randint(1,10,(2,3))
print(f"t2:{t2},shape:{t2.shape}")

t3 = torch.cat([t1,t2],dim=0) # (2,3)+(2,3)=(4,3)
print(f"t3:{t3},shape:{t3.shape}")
t4 = torch.cat([t1,t2],dim=1) # (2,3)+(2,3)=(2,6)
print(f"t4:{t4},shape:{t4.shape}")
t5 = torch.cat([t1,t2],dim=-1) #效果同上

#t6 = torch.cat([t1,t2],dim=2) #越界

print("-"*30)


#stack(),所有维度保持一致
t7 = torch.stack([t1,t2],dim=0)   #(2,3)+(2,3)=(2,2,3)
print(f"t7:{t7},shape:{t7.shape}")

t8 = torch.stack([t1,t2],dim=1)   #(2,3)+(2,3)=(2,2,3)
print(f"t8:{t8},shape:{t8.shape}")

t9 = torch.stack([t1,t2],dim=2)   #(2,3)+(2,3)=(2,3,2)
print(f"t9:{t9},shape:{t9.shape}")