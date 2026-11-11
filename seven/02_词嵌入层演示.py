"""
基于上一时刻的隐藏状态和当前时刻的隐藏状态 来预测当前时刻的输出和隐藏状态.

    ht = σ(W * ht-1 + V*x + b)
    ht : 当前时刻的隐藏状态
    σ: 激活函数, 激活函数作用: 激活隐藏状态, 激活输出.
    ht-1: 上一个时间步的输出
    W、V: 权重矩阵
    b: 偏置

    yt = σ(U * ht + b)
    yt: 当前时刻输出


案例:
    演示词嵌入层的API应用.

RNN介绍:
    全称叫Recurrent neural network, 循环神经网络, 主要处理 序列数据的.
    序列数据: 后边数据对前边的数据有依赖, 例如: 天气预测, 股市分析, 文本生成...


    组成:
        词嵌入层   
        循环网络层
        输出层

词嵌入层介绍(作用):              nn.Embedding(词向量数, 词向量维度)
    把词 (或者 词对应的索引) 转成 词向量.
"""

# 导包
import torch
import jieba            # jieba分词器, 需要安装一下, pip install jieba
import torch.nn as nn

# 1.定义函数, 用于演示 词嵌入层的API, 如何把 词(词的索引) -> 词向量
def dm01():
    # 1. 定义一句话
    text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
    # 2. 使用jiema模块进行分词.
    words = jieba.lcut(text)
    print(f'分词结果: {words}')     # ['北京', '冬奥', '的', '进度条', ...]

    # 3. 创建词嵌入层.
    # 参1: 词表大小(词的个数), 参2: 词向量的维度
    embed = nn.Embedding(len(words), 4)

    # enumerate(): 返回列表中每个值 及其对应的 索引.
    for i, word in enumerate(words):    # 效果同上.
        # print(i, word)

        # 5. 把词索引(张量形式) 转成 词向量.
        word_vector = embed(torch.tensor(i))    # 随机的, 每次都不一样, 无所谓.
        print(f'词: {word}, \t\t词向量: {word_vector}')

# 2. 测试
if __name__ == '__main__':
    dm01()