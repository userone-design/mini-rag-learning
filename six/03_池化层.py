"""
池化层作用：不会改变通道数   输入 = 输出
    1.降低参数量级
    2.降低计算复杂度
    3.降维
    nn.MaxPool2d(kernel_size=2,stride=2,padding=0) 最大池化
    nn.AvgPool2d(kernel_size=2,stride=2,padding=0) 平均池化

    

    全连接只能处理 二维 数据
"""
import torch
import torch.nn as nn

#演示单通道
def dm01():
    #创建一个1个 1通道 3乘3的矩阵
    inputs = torch.tensor([
       [ [1,2,3],
        [4,5,6],
        [7,8,9]
        ]])
    print(f"inputs:{inputs},shape:{inputs.shape}")

    #创建最大池化层
    pool1 = nn.MaxPool2d(kernel_size=2,stride=1,padding=0)
    outputs = pool1(inputs)
    print(f"outputs:{outputs},shape:{outputs.shape}")

    #创建平均池化层
    pool2 = nn.AvgPool2d(kernel_size=2,stride=1,padding=0)
    outputs = pool2(inputs)
    print(f"outputs:{outputs},shape:{outputs.shape}")



def dm02():
    #创建一个1个 3通道 3乘3的矩阵
    inputs = torch.tensor([#3通道
       [                   #1通道  3*3
        [0,1,2],
        [3,4,5],
        [6,7,8]
        ],
       [                   #2通道
        [10,20,30],
        [40,50,60],
        [70,80,90]
        ],
       [                   #3通道
        [11,22,33],
        [44,55,66],
        [77,88,99]
        ]
        ])
    print(f"inputs:{inputs},shape:{inputs.shape}")

    #创建最大池化层
    pool1 = nn.MaxPool2d(kernel_size=2,stride=1,padding=0)
    outputs = pool1(inputs)
    print(f"outputs:{outputs},shape:{outputs.shape}")

    #创建平均池化层
    pool2 = nn.AvgPool2d(kernel_size=2,stride=1,padding=0)
    outputs = pool2(inputs)
    print(f"outputs:{outputs},shape:{outputs.shape}")


    




if __name__ == '__main__':
   # dm01()
   dm02()