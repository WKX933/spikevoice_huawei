from collections import OrderedDict

import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from .SubLayers import MultiHeadAttention, PositionwiseFeedForward
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



class FFTBlock(nn.Cell):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn_s = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.slf_attn_t = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )
        self.lif_t = MultiStepLIFNode()
        self.lif_s = MultiStepLIFNode()
        self.permute = P.Transpose()
        self.reshape = P.Reshape()
        self.fill = P.MaskedFill()
        self.expand_dims = P.ExpandDims()

    def construct(self, enc_input, mask=None, slf_attn_mask=None):
        # not spike
        res = enc_input
        enc_input = self.lif_t(enc_input)
        
        ## time
        enc_input = self.permute(enc_input, (2, 1, 0, 3))
        res = self.permute(res, (2, 1, 0, 3))
        
        enc_output, enc_slf_attn = self.slf_attn_t(
            res, enc_input, enc_input, enc_input, mask=mask, mode='t'
        )
        
        ## space
        enc_output = self.permute(enc_output, (2, 1, 0, 3))
        
        if mask is not None:
            mask_expanded = self.expand_dims(mask, -1)
            enc_output = self.fill(enc_output, mask_expanded, 0)
        
        res = enc_output
        enc_output = self.lif_s(enc_output)
        
        enc_output, enc_slf_attn = self.slf_attn_s(
            res, enc_output, enc_output, enc_output, mask=mask
        )
        
        # not spike
        if mask is not None:
            mask_expanded = self.expand_dims(mask, -1)
            enc_output = self.fill(enc_output, mask_expanded, 0)
        
        # not spike
        enc_output = self.pos_ffn(enc_output)
        
        # not spike
        if mask is not None:
            mask_expanded = self.expand_dims(mask, -1)
            enc_output = self.fill(enc_output, mask_expanded, 0)

        return enc_output, enc_slf_attn


class ConvNorm(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            dilation=dilation,
            has_bias=bias,
        )

    def construct(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class PostNet(nn.Cell):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):
        super(PostNet, self).__init__()
        self.convolutions = nn.CellList()
        self.lifs = nn.CellList()

        # 第一个卷积层
        conv1 = nn.SequentialCell([
            ConvNorm(
                n_mel_channels,
                postnet_embedding_dim,
                kernel_size=postnet_kernel_size,
                stride=1,
                padding=int((postnet_kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
            ),
            nn.BatchNorm1d(postnet_embedding_dim),
        ])
        self.convolutions.append(conv1)
        self.lifs.append(MultiStepLIFNode())
        
        # 中间卷积层
        for i in range(1, postnet_n_convolutions - 1):
            conv = nn.SequentialCell([
                ConvNorm(
                    postnet_embedding_dim,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            ])
            self.convolutions.append(conv)
            self.lifs.append(MultiStepLIFNode())

        # 最后一个卷积层
        conv_last = nn.SequentialCell([
            ConvNorm(
                postnet_embedding_dim,
                n_mel_channels,
                kernel_size=postnet_kernel_size,
                stride=1,
                padding=int((postnet_kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="linear",
            ),
            nn.BatchNorm1d(n_mel_channels),
        ])
        self.convolutions.append(conv_last)
        
        self.lif0 = MultiStepLIFNode()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, x):
        x = self.lif0(x)
        
        # x = x.contiguous().transpose(2, 3)
        x = self.transpose(x, (0, 1, 3, 2))
        
        # 获取形状
        shape = self.shape(x)
        T, B, D, L = shape[0], shape[1], shape[2], shape[3]
        
        # 前n-1个卷积层
        for i in range(len(self.convolutions) - 1):
            # 展平处理: (T, B, D, L) -> (T*B, D, L)
            x_flat = self.reshape(x, (T * B, D, L))
            # 卷积
            conv_out = self.convolutions[i](x_flat)
            # 恢复形状: (T*B, D, L) -> (T, B, D, L)
            x = self.reshape(conv_out, (T, B, -1, L))
            # LIF激活
            x = self.lifs[i](x)
            # 更新D维度
            D = x.shape[2]
        
        # 最后一个卷积层
        x_flat = self.reshape(x, (T * B, D, L))
        x = self.convolutions[-1](x_flat)
        x = self.reshape(x, (T, B, -1, L))
        
        # x = x.contiguous().transpose(2, 3)
        x = self.transpose(x, (0, 1, 3, 2))
        
        return x