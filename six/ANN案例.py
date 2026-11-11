"""
ANN人工神经网络：手机价格分类
背景：
    基于手机的20列特征 预测手机的价格区间 

ANN实例jie
    1.构建数据集
    2.搭建神经网络
    3.模型训练
    4.模型测试
优化方法：
     1.优化方法SGD->Adam
     2.学习率->小学习率
     3.对数据标准化
     4.训练的轮数
     5.增加网络的深度 每层的神经元数


"""
import torch
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch import nn
from torch import optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from torchsummary import summary
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False


# 1.构建数据集
def create_dataset():
    #1.加载数据集
    data = pd.read_csv('./data/手机价格预测.csv')
    # print(f"data:{data}")
    # print(f"data.shape:{data.shape}")
    #2.获取x特征和y标签
    #:-1不要最后一个 前面全要   ,-1只要最后一个
    x,y = data.iloc[:,:-1],data.iloc[:,-1]
    # print(f"x:{x.head()},{x.shape}")
    # print(f"y:{y.head()},{y.shape}")
    #3.把特征列转化为浮点型
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    #4.数据集划分  
    #参数1：数据集 2：标签 3：训练集占比 4：随机种子 5：样本的分类（参考y的类别进行抽取数据）
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=88,stratify=y)
    #优化 数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    train_dataset = TensorDataset(torch.from_numpy(x_train),torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.from_numpy(x_test),torch.tensor(y_test.values))
    print(f"train_dataset:{train_dataset} test_dataset:{test_dataset}")
    #                                 #20 充当输入特征数     4 充当输出特征数
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))   #np.unique去重

# 2.搭建神经网络
#三个全连接层 1 输入维度20 输出维度128  2 输入维度128 输出维度256  3 输入维度256 输出维度4
class PhonePriceModel(nn.Module):
    #1.初始化父类成员
    def __init__(self,input_dim,output_dim):
        #c初始化父类成员
        super(PhonePriceModel,self).__init__()
        #2.定义成员变量
        #隐藏层1
        self.fc1 = nn.Linear(input_dim,128)
        #隐藏层2
        self.fc2 = nn.Linear(128,256)
        #隐藏层3
        self.fc3 = nn.Linear(256,512)
        #隐藏层4
        self.fc4 = nn.Linear(512,128)
        self.output = nn.Linear(128,output_dim)
    def forward (self,x):
        #隐藏层1
        x = torch.relu(self.fc1(x))
        #隐藏层2
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        #输出层: 加权求和 + 激活函数(softmax)  -> 这里只需要做 加权求和.
        # 正常写法, 但是不需要, 后续用 多分类交叉熵损失函数 CrossEntropyLoss() 替代
        # CrossEntropyLoss() = softmax() + 损失计算
        # x = torch.softmax(self.output(x), dim=1)
        x = self.output(x)
        # 2.4 返回处理结果
        return x
# 3.模型训练
def train(train_dataset,input_dim,output_dim):
    #1.创建数据加载器 流程：数据 -> 张量 -> 数据集 -> 数据加载器
    # 参数1：数据集1600条 参数2 ：每批次的数据条数 参3：是否打乱数据
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    # 2.创建模型
    model = PhonePriceModel(input_dim,output_dim)
    #3.定义损失函数 多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #4.创建优化器对象
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    #5.训练
   # 5. 模型训练.
    # 5.1 定义变量, 记录训练的 总轮数.
    epochs = 50
    # 5.2 开始(每轮的)训练.
    for epoch in range(epochs):
        # 5.2.1 定义变量, 记录每次训练的损失值, 训练批次数.
        total_loss, batch_num = 0.0, 0
        # 5.2.2 定义变量, 表示训练开始的时间.
        start = time.time()
        # 5.2.3 开始本轮的 各个批次的训练.
        for x,y in train_loader:
            # 5.2.4 切换模型(状态)
            model.train()#训练模式 .eval 测试模式
            #模型预测
            y_pred = model(x)
            #计算损失
            loss = criterion(y_pred,y)
            #梯度清零 +反向传播+梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #累计损失
            total_loss += loss.item()#把本轮的每批次16条的平均损失累计起来
            batch_num += 1
        #打印训练信息
        print(f"轮数：{epoch+1},平均损失值：{total_loss/batch_num:.4f},time:{time.time()-start:.4f}")
    #6.多轮训练结束 保存模型
    #参数1 模型对象的参数 参数2 保存的文件名
    #print(f"\n\n模型的参数信息：{model.state_dict()}\n\n")
    torch.save(model.state_dict(),'./model/phone.pth')

#模型测试
def evaluate(test_dataset,input_dim,output_dim):
    #1.创建神经网络分类对象
    model = PhonePriceModel(input_dim,output_dim)
    #2.加载模型参数
    model.load_state_dict(torch.load('./model/phone.pth'))
    #3.创建测试集数据加载器 参1 数据集对象400条测试集
    test_loader = DataLoader(test_dataset,batch_size=8,shuffle=False)
    #4.定义变量 记录预测正确的样本个数
    correct = 0
    #5.从数据加载中 获取到每批次的数据
    for x,y in test_loader:
        #训练模式->预测模式
        model.eval()
        y_pred = model(x)
        #print(f"y_pred:{y_pred}")
        #根据加权求和 得到类别 用argmax获取最大值对应下标,就是类别
        y_pred = torch.argmax(y_pred,dim=1)#dim=1表示逐行处理
        print(f"预测结果：{y_pred}")
        # 5.4 统计预测正确的样本个数.
        print(y_pred==y)
        print((y_pred == y).sum())
        correct += (y_pred == y).sum().item()
    print(f"预测正确的样本个数：{correct / len(test_dataset):.4f}")



if __name__ == '__main__':
    #1.数据集的创建
    train_dataset,test_dataset,input_dim,output_dim = create_dataset()
    # print(f"train_dataset:{train_dataset}")
    # print(f"test_dataset:{test_dataset}")
    # print(f"input_dim:{input_dim}") #20
    # print(f"output_dim:{output_dim}") #4
    #2.搭建神经网络.构建模型
    #model =PhonePriceModel(input_dim,output_dim)
    #参数2：输入数据的形状(批次大小，输入特征数) 每批16条 每条20列特征
    #summary(model,(16,input_dim))
    #3.模型训练
    train(train_dataset,input_dim,output_dim)
    #4.模型测试
    evaluate(test_dataset,input_dim,output_dim)