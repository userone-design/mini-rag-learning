"""
目的：避免模型过拟合
L1：权重可降为0 = 降维
l2：权重可趋近于0 

Dropout：随机丢弃部分神经元，防止过拟合，测试过程中不起作用
一批次一批次训练，每个批次的训练数据中，死亡的神经元不确定通过组合数据，增加数据量
 未被杀死的权重p = 1/(1-p) p通常取0.2-0.5
"""

import torch
import torch.nn as nn   
def dm01():
    #1.创建隐藏层输出结果
    t1 = torch.randint(0, 10, (1, 4)).float()
    print("t1: {t1}")
    #.进行下一层加权计算 和激活函数计算
    #参1：输入维度 参2：输出维度
    linear1 = nn.Linear(4, 5)

    #2.加权求和
    l1 = linear1(t1)
    print(f'加权求和:{l1}')

    #3.激活函数
    output = torch.relu(l1)  #relu激活函数会把负数变为0
    print(f"激活函数：{output}")

    #4.对激活值进行随机失活
    dropout = nn.Dropout(p=0.5)
    d1 = dropout(output)
    print(f"随机失活：{d1}")









if __name__ == '__main__':
    dm01()