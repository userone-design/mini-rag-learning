"""
近30天 天气分布情况

结论：
  针对B(调节权重系数)，值越大说明月以来指数加权平均，越不依赖自己的梯度值
  数据就越平缓

  梯度下降优化方法————指数加权平均
不是梯度下降优化方式
而是后续要学习的 动量法Momentum,自适应学习率:AdaGrad,RMSProp
自适应动量法:Adam,这四种方式底层公式要用到：指数加权平均
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
ELEMENT_NUMBER = 30


#1.不考虑权重系数
def dm01():
    torch.manual_seed(0)
    temperature = torch.randn([ELEMENT_NUMBER,])*10
    print(temperature)
    days = torch.arange(1,ELEMENT_NUMBER+1,1)
    plt.plot(days,temperature,color='r')
    plt.scatter(days,temperature)
    plt.show()


#2.指数加权平均
def dm02(beta=0.9):
    torch.manual_seed(0)
    temperature = torch.randn([ELEMENT_NUMBER,])*10
    print(temperature)
    exp_weight_avg = []
    for idx,temp in enumerate(temperature,1):
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        new_temp = exp_weight_avg[idx-2]*beta+(1-beta)*temp
        exp_weight_avg.append(new_temp)
    days = torch.arange(1,ELEMENT_NUMBER+1,1)
    plt.plot(days,exp_weight_avg,color='r')
    plt.scatter(days,temperature)
    plt.show()

if __name__ == "__main__":
   # dm01()
    dm02(0.9)