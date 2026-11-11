"""
梯度下降思想：使损失函数最小的方法
w新=w旧-学习率*梯度
epoch：训练轮数
batch size：一批的数据量
iter：批次的数量=迭代器数量  batch=iter 

100轮 总100条 每批16条
每轮7批：16 16 16 16 16 16 4  总700批 700个iter

梯度下降几种方式区别就是Batch Size不同
        Training Set Size       Batch Size    Number of Batches
BGD              N                 N                1
SGD              N                 1                N
Min-Batch        N                 B            (N+B-1)//B

反向传播：利用损失函数 从后往前 结合梯度下降算法，依次求偏导，进行参数更新
 
梯度下降优化方法————指数加权平均
不是梯度下降优化方式
而是后续要学习的 动量法Momentum,自适应学习率:AdaGrad,RMSProp
自适应动量法：Adam,这四种方式底层公式要用到：指数加权平均
"""