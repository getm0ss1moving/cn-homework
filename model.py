"""
model.py
轻量级 CNN 模型定义 —— 用于"基于 TCP 通信的联邦学习系统"(MNIST 手写数字分类)

说明:
- Server 端与 Client 端都需要 import 本文件, 用来实例化结构完全一致的模型。
- 联邦学习中, 网络上真正传输的是 model.state_dict()(权重参数),
  使用 pickle / io.BytesIO 序列化为字节流后, 再通过 TCP 发送。
- 本模型参数约 20.7 万个, state_dict 序列化后约 0.8 MB(远超单次 recv(1024) 的范围),
  因此必须按照"先发 8 字节长度, 再 while 循环 recv 拼接"的方式来解决 TCP 粘包问题。
- 若改用 CIFAR-10 等彩色数据集, 只需把 conv1 的输入通道由 1 改为 3 即可。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """轻量级卷积神经网络 (输入 1x28x28, 输出 10 类)"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # ---- 卷积特征提取 ----
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)    # 1x28x28  -> 16x28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # 16x14x14 -> 32x14x14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)          # 每次尺寸减半
        # ---- 全连接分类 ----
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # -> 16x14x14
        x = self.pool(F.relu(self.conv2(x)))   # -> 32x7x7
        x = torch.flatten(x, 1)                # -> 1568
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_model(num_classes: int = 10) -> "CNN":
    """工厂函数: 返回一个新的模型实例, 方便 Server/Client 统一调用。"""
    return CNN(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 自检: 打印参数量与 state_dict 序列化后的字节大小,
    # 这些数字可以直接写进实验报告的"TCP 封包结构定义"一节。
    import io
    import pickle

    model = build_model()
    n_params = count_parameters(model)

    buffer = io.BytesIO()
    pickle.dump(model.state_dict(), buffer)
    size_bytes = buffer.tell()

    # 验证前向传播形状
    dummy = torch.randn(4, 1, 28, 28)
    out = model(dummy)

    print("=" * 56)
    print(f"模型参数总数          : {n_params:,}")
    print(f"state_dict 序列化大小 : {size_bytes:,} 字节  约 {size_bytes / 1024:.1f} KB")
    print(f"前向输出形状          : {tuple(out.shape)}  (应为 (4, 10))")
    print(f"单次 recv(1024) 收完?  : {'否, 必须 while 循环 recv 拼接' if size_bytes > 1024 else '是'}")
    print("=" * 56)
