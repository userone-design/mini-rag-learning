"""
half()[是float16],
double()[是float64],
float()[是float32],
short(),
int()[是int32],
long()[是int64]
等方法可以将张量转换为不同的数据类型。
掌握：
type(torch.int32)方法可以将张量转换为不同的数据类型。
"""
import torch
#直接创建指定类型
t1 = torch.tensor([1,2,3,4,5],dtype=torch.float32) # 指定类型
print(f"t1:{t1},type:{{type(t1)}},(元素)类型:{t1.dtype}")
print("-" * 30)

#将张量转换为不同的数据类型
t2 = t1.type(torch.int32) # 转换为int32类型
print(f"t2:{t2},type:{{type(t2)}},(元素)类型:{t2.dtype}")
print("-" * 30)

print(t2.half())