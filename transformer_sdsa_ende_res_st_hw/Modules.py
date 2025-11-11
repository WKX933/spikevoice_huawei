import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class MultiSpike4(nn.Cell):
    """
    Quantized spike function: clamp to [0, 4] and round.
    """

    def __init__(self):
        super(MultiSpike4, self).__init__()
        self.clip = ops.clip_by_value
        self.round = ops.Rint()  # same as torch.round()
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()

    def construct(self, x):
        # quantization (no ctx, so use surrogate gradient)
        x_clamped = self.clip(x, 0.0, 4.0)
        x_quant = self.round(x_clamped)
        return x_quant

    # 如果需要近似梯度，可自定义反向传播
    def bprop(self, x, out, dout):
        grad_input = dout.copy()
        grad_input = ops.where(x < 0, ms.Tensor(0.0, ms.float32), grad_input)
        grad_input = ops.where(x > 4, ms.Tensor(0.0, ms.float32), grad_input)
        return (grad_input,)


class MultiStepLIFNode(nn.Cell):
    """
    Multi-step Leaky Integrate-and-Fire node
    with quantized spike activation.
    """

    def __init__(self, decay=0.2, act=False):
        super(MultiStepLIFNode, self).__init__()
        self.decay = ms.Tensor(decay, ms.float32)
        self.act = act
        self.qtrick = MultiSpike4()
        self.zeros_like = ops.ZerosLike()

    def construct(self, x):
        # x: (T, B, ..., hidden_dim)
        T = x.shape[0]

        mem = self.zeros_like(x[0])
        spike = self.zeros_like(x[0])
        output = ops.zeros_like(x)

        for t in range(T):
            if t > 0:
                mem = (mem - ops.stop_gradient(spike)) * self.decay + x[t]
            else:
                mem = x[t]
            spike = self.qtrick(mem)
            mem_old = mem
            output[t] = spike

        return output



class ScaledDotProductAttention(nn.Cell):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(axis=2)
        self.lif_kv = MultiStepLIFNode()
        
        # 定义操作
        self.mul = P.Mul()
        self.sum = P.ReduceSum(keep_dims=True)
        self.unsqueeze = P.ExpandDims()
        self.transpose = P.Transpose()
        self.masked_fill = P.MaskedFill()
        self.shape = P.Shape()

    def construct(self, q, k, v, mask=None, mode='s'):
        mask = None
        T, B, L, _ = self.shape(q)
        
        # 计算注意力
        attn = self.mul(k, v)

        if mask is not None and mode == 's':
            # attn = attn.masked_fill(mask.unsqueeze(-1), 0)
            mask_expanded = self.unsqueeze(mask, -1)
            attn = self.masked_fill(attn, mask_expanded, 0)
            
        if mask is not None and mode == 't':
            # attn = attn.masked_fill(mask.transpose(0,2).unsqueeze(-1), 0)
            mask_transposed = self.transpose(mask, (0, 2, 1))
            mask_expanded = self.unsqueeze(mask_transposed, -1)
            attn = self.masked_fill(attn, mask_expanded, 0)

        # 计算键值聚合
        kv = self.sum(attn, -2)  # sum(dim=-2, keepdim=True)
        kv = self.lif_kv(kv)
        
        # 计算输出
        output = self.mul(q, kv)

        return output, attn