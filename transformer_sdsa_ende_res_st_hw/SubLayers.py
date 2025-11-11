import mindspore.nn as nn
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from .Modules import ScaledDotProductAttention
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


class MultiHeadAttention(nn.Cell):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Dense(d_model, n_head * d_k)
        self.w_ks = nn.Dense(d_model, n_head * d_k)
        self.w_vs = nn.Dense(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm([d_model])

        self.fc = nn.Dense(n_head * d_v, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.lif_1 = MultiStepLIFNode()
        self.lif_q = MultiStepLIFNode()
        self.lif_k = MultiStepLIFNode()
        self.lif_v = MultiStepLIFNode()
        self.batch_norm1 = nn.BatchNorm1d(n_head * d_k)
        self.batch_norm2 = nn.BatchNorm1d(n_head * d_k)
        self.batch_norm3 = nn.BatchNorm1d(n_head * d_v)
        
        # 定义操作
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.permute = P.Transpose()
        self.repeat = P.Tile()
        self.shape = P.Shape()
        self.flatten = P.Reshape()

    def construct(self, res, q, k, v, mask=None, mode='s'):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        T, sz_b, len_q, _ = self.shape(q)
        T, sz_b, len_k, _ = self.shape(k)
        T, sz_b, len_v, _ = self.shape(v)
        
        # q,k,v should be spike
        residual = res

        # 线性变换并展平
        q = self.flatten(self.w_qs(q), (T * sz_b, len_q, -1))
        k = self.flatten(self.w_ks(k), (T * sz_b, len_k, -1))
        v = self.flatten(self.w_vs(v), (T * sz_b, len_v, -1))
        # (96,133,256)
        
        # 批归一化
        k = self.transpose(k, (0, 2, 1))
        k = self.batch_norm1(k)
        k = self.transpose(k, (0, 2, 1))
        k = self.reshape(k, (T, sz_b, len_k, -1))
        
        q = self.transpose(q, (0, 2, 1))
        q = self.batch_norm2(q)
        q = self.transpose(q, (0, 2, 1))
        q = self.reshape(q, (T, sz_b, len_q, -1))
        
        v = self.transpose(v, (0, 2, 1))
        v = self.batch_norm3(v)
        v = self.transpose(v, (0, 2, 1))
        v = self.reshape(v, (T, sz_b, len_v, -1))
        # (4,24,132,256)
        
        # 重塑为多头
        k = self.reshape(k, (T, sz_b, len_k, n_head, d_k))
        v = self.reshape(v, (T, sz_b, len_v, n_head, d_v))
        q = self.reshape(q, (T, sz_b, len_q, n_head, d_k))
        # (4,12,132,2,128)
        
        # 调整维度顺序
        q = self.reshape(self.permute(q, (0, 3, 1, 2, 4)), (T, -1, len_q, d_k))  # (n*b) x lq x dk
        k = self.reshape(self.permute(k, (0, 3, 1, 2, 4)), (T, -1, len_k, d_k))  # (n*b) x lk x dk
        v = self.reshape(self.permute(v, (0, 3, 1, 2, 4)), (T, -1, len_v, d_v))  # (n*b) x lv x dv
        
        # LIF激活
        q = self.lif_q(q)
        k = self.lif_k(k)
        v = self.lif_v(v)

        # 扩展掩码
        if mask is not None:
            mask = self.repeat(mask, (1, n_head, 1))  # (n*b) x .. x ..
        
        # 注意力计算
        output, attn = self.attention(q, k, v, mask=mask, mode=mode)

        # 重塑输出
        output = self.reshape(output, (T, n_head, sz_b, len_q, d_v))
        output = self.reshape(
            self.permute(output, (0, 2, 3, 1, 4)), (T, sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        # 最终线性变换和层归一化
        # output = self.dropout(self.fc(output))
        output = self.fc(output)
        # output = self.lif_1(output)
        output = self.layer_norm(output + residual)
        # output = output + residual
        
        return output, attn


class PositionwiseFeedForward(nn.Cell):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            pad_mode='pad',
            padding=(kernel_size[0] - 1) // 2,
            has_bias=True
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            pad_mode='pad',
            padding=(kernel_size[1] - 1) // 2,
            has_bias=True
        )

        self.layer_norm = nn.LayerNorm([d_in])
        self.dropout = nn.Dropout(p=dropout)
        self.lif_0 = MultiStepLIFNode()
        self.lif_1 = MultiStepLIFNode()
        self.lif_2 = MultiStepLIFNode()
        
        # 定义操作
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, x):
        residual = x
        # residual should be spike
        # (4,24,117,256)
        x = self.lif_0(x)
        output = self.transpose(x, (0, 1, 3, 2))
        T, B, D, L = self.shape(output)
        
        # 展平批次和时间维度
        output = self.reshape(output, (T * B, D, L))
        # (96,256,117)
        
        # 第一层卷积 + LIF
        conv1_out = self.w_1(output)
        conv1_out_reshaped = self.reshape(conv1_out, (T, B, -1, L))
        lif1_out = self.lif_1(conv1_out_reshaped)
        lif1_out_flat = self.reshape(lif1_out, (T * B, -1, L))
        
        # 第二层卷积
        output = self.w_2(lif1_out_flat)
        # output = self.lif_2(output)
        
        # 恢复形状
        output = self.reshape(output, (T, B, -1, L))
        output = self.transpose(output, (0, 1, 3, 2))
        # output = self.dropout(output)
        output = self.layer_norm(output + residual)
        # output = output + residual

        return output