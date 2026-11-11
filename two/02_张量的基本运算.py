"""
add() sub() mul() div() neg()取反符号 +-
add()_sub()_ mul()_ div()_ neg()_可以修改源数据
 +-*/ 可代替函数

张量和数值运算 运算会运算到每一个值上
"""
import torch
import numpy

t1 = torch.tensor([1,2,3])
print(f"t1:{t1},type:{type(t1)}")
#t2 = t1.add(10)
#print(f"t2:{t2},type:{type(t2)}")

# t1 +=10
# print(f"t1:{t1},type:{type(t1)}")

# t1 -=2
# print(f"t1:{t1},type:{type(t1)}")
# t2 = t1.neg()
# print(f"t2:{t2},type:{type(t2)}")
t2 = t1/2 #不能整除
t3 = t1//2 #能整除
print(f"t2:{t2},type:{type(t2)}")
print(f"t3:{t3},type:{type(t3)}")

 
