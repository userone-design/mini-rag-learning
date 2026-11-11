"""
两个张量的维度保持一致 
t1 * t2
t3 = t1.mul(t2) 同行同列

t1 @ t2
    #A列=B行
t3 = t1.matmul(t2)  

t3 = t1.dot(t2) 只在一维里有用
"""
import torch
def dm01():
    #同行同列
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    t2 = torch.tensor([[1,2,3],[4,5,6]])
    t3 = t1 * t2
#    t3 = t1.mul(t2)#效果同上
    print(f"t3:{t3}") 

def dm02():
    #A列=B行
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    t2 = torch.tensor([[1,2],[3,4],[5,6]])
    t3 = t1 @ t2
#   t3 = t1.matmul(t2)#效果同上
    print(f"t3:{t3}") 

def dm03():
    t1 = torch.tensor([1,2,3])
    t2 = torch.tensor([4,5,6])
    t3 = t1.dot(t2)
    print(f"t3:{t3}")

if __name__ == '__main__':
 #   dm01()
#   dm02()
    dm03()
