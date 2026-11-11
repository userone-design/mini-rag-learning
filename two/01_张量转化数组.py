"""
1 张量->numpy ndarray
    张量对象.numpy()共享内存√
    张量对象.numpy().copy() 不共享内存

2 numpy ndarray->张量
   torch.tensor(nd数组) 不共享√
   from_numpy() 共享

3 从标量张量中提取内容
  标量张量.item()√
"""
import torch
import numpy as np
def dm01():
    t1 = torch.tensor([1,2,3])
    t2 = t1.numpy().copy()
    print(f"t1:{t1},type{type(t1)}")
    print(f"t2:{t2},type:{type(t2)}")

def dm02():
    t1 = np.array([1,2,3])
    t2 = torch.tensor(t1).type(torch.float32)
    print(f"t1;{t1},type:{type(t1)}")
    print(f"t2;{t2},type:{type(t2)}")

def dm03():
    t1 = torch.tensor(100)
    t2 = t1.item() #提取 100 是int型，提取元素，并且只能提取一个值，标量张量
    print(f"t1;{t1},type:{type(t1)}")
    print(f"t2;{t2},type:{type(t2)}")


if __name__ == '__main__':
    # dm01()
    dm02()
   # dm03()
