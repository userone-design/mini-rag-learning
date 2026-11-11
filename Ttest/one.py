# 以开源的 tiktoken 库为例（OpenAI 的分词器）
# 先安装：pip install tiktoken
import tiktoken

# 加载分词器模型

# enc = tiktoken.get_encoding("cl100k_base")

# text = "多模态检索技术是什么?Hello world!"
# tokens = enc.encode(text)  # 变成数字 ID
# print(f"数字ID: {tokens}")

# # 把数字还原成文字块，看看它是怎么切的
# print(f"切分结果: {[enc.decode([t]) for t in tokens]}")

import math
import torch
from torch import Tensor, nn


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape

        q = self.q_proj(x)  # [B, S, H]
        k = self.k_proj(x)  # [B, S, H]
        v = self.v_proj(x)  # [B, S, H]

        # 只转置最后两个维度：
        # [B, S, H] -> [B, H, S]
        k_t = k.transpose(-2, -1)

        # [B, S, H] @ [B, H, S] -> [B, S, S]
        scores = q @ k_t

        # 单头情况下 d_k = hidden_dim
        scaled_scores = scores / math.sqrt(hidden_dim)

        # 每一行对所有 key 做 softmax
        attention_weights = torch.softmax(scaled_scores, dim=-1)

        # [B, S, S] @ [B, S, H] -> [B, S, H]
        output = attention_weights @ v

        assert output.shape == x.shape
        return output

