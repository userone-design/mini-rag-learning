import math
import torch
from torch import Tensor, nn

class MultiHeadSelfAttention(nn.Module): # 改名：这其实是多头
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads # 补全：必须定义头数
        self.head_dim = hidden_dim // num_heads # 补全：每个头的维度
        
        # 确保能整除
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim 必须能被 num_heads 整除"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 补全：最后的输出投影层
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        
        # 1. 线性映射
        q_flat = self.q_proj(x)  # [B, S, H]
        k_flat = self.k_proj(x)
        v_flat = self.v_proj(x)

        # 2. 拆分成多头并转置 (注意缩进修复)
        q = q_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, S, head_dim]
        k = k_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数
        k_t = k.transpose(-2, -1)  # [B, heads, head_dim, S]
        scores = q @ k_t           # [B, heads, S, S]
        scores = scores / math.sqrt(self.head_dim)

        # 4. Softmax 计算权重并加权求和
        attention_weights = torch.softmax(scores, dim=-1)
        head_outputs = attention_weights @ v  # [B, heads, S, head_dim]

        # 5. 拼接多个 head 回 [B, S, H]
        concatenated = (
            head_outputs.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, self.hidden_dim)
        )

        # 6. 混合不同 head 的信息
        output = self.out_proj(concatenated)
        
        return output # 补全：加上 return

if __name__ == '__main__':
    batch_size = 2
    seq_len = 5
    hidden_dim = 512
    num_heads = 8

    # 模拟输入数据
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 实例化模型
    mha = MultiHeadSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads)
    
    # 前向传播
    output = mha(x)
    
    print(f"输入形状: {x.shape}")       # [2, 5, 512]
    print(f"输出形状: {output.shape}")  # [2, 5, 512] -> 完美一致！