"""
张量->存储同一类型元素的数据，必须是数值才可以

1.张量的创建方式：
   torch.tensor 小写根据指定数创建张量，可在创建元素时指定类型
   torch.Tensor 大写根据形状（2行3列）创建张量

   torch.intTensor()  # int32
   torch.floatTensor() # float32
2.创建线性和随机张量
   arange() linspace()
   rand() randn() randint()

3.创建0、1、指定张量
   torch.ones()  ones_like()  torch.zeros() _like 
   torch.full((2,3),255)

4.张量类型转化
   type(torch,int32)
   





torch.tensor 小写根据指定数创建张量，可在创建元素时指定类型
torch.Tensor 大写根据形状（2行3列）创建张量,不可指定类型，默认float32

torch.tensor(data,dtype=torch.float32) # 指定类型
"""
import torch
import numpy as np

def dm01():
    #标量 张量
    t1 = torch.tensor(10)
    print(f"t1:{t1},type:{type(t1)}")
    print("-" * 30)

    #二维列表——>张量
    data = ([1,2,3],[4,5,6])
    t2 = torch.tensor(data)
    print(f"t2:{t2},type:{type(t2)}")
    print("-" *  30)
   

    #numpy nd数组——>张量
    data = np.random.randint(0,10,(2,3))
#    t3 = torch.tensor(data)
    t3 = torch.tensor(data,dtype=torch.float)
#   t3 = torch.tensor(data,dtype=torch.float32) # 指定类型
#    print(f"t3:{t3},type:{type(t3)}")

if __name__ == '__main__':
    dm01()