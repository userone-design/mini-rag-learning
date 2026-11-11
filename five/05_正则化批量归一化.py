"""

对数据先分批，对每批数据先标准化再进行缩放平移。

先对数据标准化(均值为0 标准差为1)，再对数据缩放+平移
     丢失一些信息                    找回一些信息

例如:使用sigmmoid激活函数,他只有在[-3,3]有较好的效果
如果没有标准化(数据没有在上述区间)，可能模型收敛较慢，标准化之后，数据在[0,1]之间，分布更均匀，模型收敛更好

应用场景:批量归一化在计算机视觉广泛使用

1.正则化的作用：避免过拟合
2.批量归一化的作用：防止梯度消失和梯度爆炸
3.随机失活Dropout：避免过拟合 
4.BN层是什么：毒素据标准化，再对数据重构  affine=True


"""

import torch
import torch.nn as nn

#1.定义函数 处理二维数据
#一张图片，2通道，3行4列

def dm01():
     input_2d = torch.randn(1,2,3,4)
     print(f"input_2d：{input_2d}")
     #2.创建批量归一化层 
     # #参数：通道数，小常数， 动量值用于计算移动平均统计量 ，是否可学习
     bn2d = nn.BatchNorm2d(num_features=2,eps=1e-5,momentum=0.1,affine=True)
     output_2d = bn2d(input_2d)
     print(f"output_2d：{output_2d}")

def dm02():
     inpurt_1d = torch.randn(2,2)
     print(f"inpurt_1d：{inpurt_1d}")
     #创建线性层
     linear1 = nn.Linear(2,4)
     l1 = linear1(inpurt_1d)
     print(f"l1：{l1}")
     #创建批量归一化层
     bn1d = nn.BatchNorm1d(num_features=2)
     output_1d = bn1d(inpurt_1d)
     print(f"output_1d：{output_1d}")



if __name__ == '__main__':
   # dm01()
    dm02()
