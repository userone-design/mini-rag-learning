"""
arange(),linspace(),manual_seed(),randint()函数的使用
"""
import torch
def dm02():
    # 01 指定范围的 线性张量
    t1 = torch.arange(0,10,2) # 0-10之间，步长为2
    print(f"t1:{t1},type:{type(t1)}")
    print("-" * 30)
    # 02 指定范围的 线性张量 -> 等差数列
    t2 = torch.linspace(1,10,4)# 1-10之间，个数为4
    print(f"t2:{t2},type:{type(t2)}")
    print("-" * 30)

def dm03():
    #设置随机种子 全局都可以用 如果写 随机的将会固定
    torch.manual_seed(2)

    # 01 均匀分布的（0，1）随机张量
    t3 = torch.rand(2, 3) # 2行3列
    print(f"t3:{t3},type:{type(t3)}")
    print("-" * 30)
    #02 符合正态分布的随机张量
    t4 = torch.randn(2, 3) # 2行3列
    print(f"t4:{t4},type:{type(t4)}")
    print("-" * 30)
    # 03 指定范围的随机张量
    t5 = torch.randint(1, 10, (3, 5)) # 2行3列，范围是0-10
    print(f"t5:{t5},type:{type(t5)}")
    print("-" * 30)

if __name__ == '__main__':
 #   dm02()
    dm03()