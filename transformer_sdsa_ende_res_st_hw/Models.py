import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import functional as F

import transformer.Constants as Constants
from .Layers import FFTBlock
# from transformer.Layers import FFTBlock as fft_dec
from text.symbols import symbols
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

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return mindspore.Tensor(sinusoid_table, dtype=mindspore.float32)


class Encoder(nn.Cell):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        self.T = config["T"]
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        
        # 使用Parameter保存位置编码
        position_enc_table = get_sinusoid_encoding_table(n_position, d_word_vec)
        self.position_enc = mindspore.Parameter(
            F.expand_dims(position_enc_table, 0), 
            requires_grad=False
        )

        self.time_embed = mindspore.Parameter(mindspore.ops.Zeros()((1, self.T, d_word_vec), mindspore.float32))

        self.layer_stack = nn.CellList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        
        # 定义操作
        self.unsqueeze = P.ExpandDims()
        self.repeat = P.Tile()
        self.permute = P.Transpose()
        self.shape = P.Shape()
        self.expand = P.BroadcastTo

    def construct(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        # src_seq = (src_seq.unsqueeze(0)).repeat(self.T, 1, 1)
        src_seq = self.repeat(self.unsqueeze(src_seq, 0), (self.T, 1, 1))
        # (4,12,124)
        T, batch_size, max_len = self.shape(src_seq)

        # -- Prepare masks
        # mask = mask.unsqueeze(0).repeat(T,1,1)
        mask = self.repeat(self.unsqueeze(mask, 0), (T, 1, 1))
        # slf_attn_mask = mask.unsqueeze(2).expand(-1,-1, max_len, -1)
        slf_attn_mask = self.expand((T, batch_size, max_len, max_len))(self.unsqueeze(mask, 2))

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            print('length of src larger than max_length!!')
            enc_output = self.src_word_emb(src_seq)
            pos_enc = get_sinusoid_encoding_table(
                src_seq.shape[2], self.d_model
            )[: src_seq.shape[2], :]
            pos_enc = self.repeat(self.unsqueeze(pos_enc, 0), (batch_size, 1, 1))
            enc_output = enc_output + pos_enc
        else:
            enc_output = self.src_word_emb(src_seq)
            position_enc_slice = self.position_enc[:, :max_len, :]
            position_enc_expanded = self.repeat(position_enc_slice, (batch_size, 1, 1))
            enc_output = enc_output + position_enc_expanded

        ### time embedding
        # enc_output=enc_output.permute(2,1,0,3).contiguous()
        enc_output = self.permute(enc_output, (2, 1, 0, 3))
        time_embed_expanded = self.repeat(self.time_embed, (batch_size, 1, 1))
        enc_output = enc_output + time_embed_expanded
        # enc_output=enc_output.permute(2,1,0,3).contiguous()
        enc_output = self.permute(enc_output, (2, 1, 0, 3))

        # enc_output = self.atan(enc_output)
        for enc_layer in self.layer_stack:
            # not spike
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Cell):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]
        self.T = config["T"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        # 使用Parameter保存位置编码
        position_enc_table = get_sinusoid_encoding_table(n_position, d_word_vec)
        self.position_enc = mindspore.Parameter(
            F.expand_dims(position_enc_table, 0), 
            requires_grad=False
        )
        
        self.time_embed = mindspore.Parameter(mindspore.ops.Zeros()((1, self.T, d_word_vec), mindspore.float32))

        self.layer_stack = nn.CellList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        
        # 定义操作
        self.unsqueeze = P.ExpandDims()
        self.repeat = P.Tile()
        self.permute = P.Transpose()
        self.shape = P.Shape()
        self.expand = P.BroadcastTo
        self.min = P.Minimum()

    def construct(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        # enc_seq(24,864,256)
        T, batch_size, max_len, D = self.shape(enc_seq)

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            print('seq_len out of range!!!!!!!!!!!!!')
            # slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            slf_attn_mask = self.expand((batch_size, max_len, max_len))(self.unsqueeze(mask, 1))
            dec_output = enc_seq
            pos_enc = get_sinusoid_encoding_table(
                enc_seq.shape[2], self.d_model
            )[: enc_seq.shape[2], :]
            pos_enc = self.repeat(self.unsqueeze(pos_enc, 0), (batch_size, 1, 1))
            dec_output = dec_output + pos_enc
        else:
            max_len = self.min(max_len, self.max_seq_len)

            # -- Prepare masks
            # mask(24,866)
            # mask = mask.unsqueeze(0).repeat(T,1,1)
            if mask is not None:
                mask = self.repeat(self.unsqueeze(mask, 0), (T, 1, 1))
                # slf_attn_mask = mask.unsqueeze(2).expand(-1,-1, max_len, -1)
                slf_attn_mask = self.expand((T, batch_size, max_len, max_len))(self.unsqueeze(mask[:, :, :max_len], 2))
                mask = mask[:, :, :max_len]
                slf_attn_mask = slf_attn_mask[:, :, :, :max_len]
            else:
                slf_attn_mask = None            
            # enc_seq(24,866,256)
            enc_seq_slice = enc_seq[:, :, :max_len, :]
            position_enc_slice = self.position_enc[:, :max_len, :]
            position_enc_expanded = self.repeat(position_enc_slice, (batch_size, 1, 1))
            dec_output = enc_seq_slice + position_enc_expanded

            # dec_output=dec_output.permute(2,1,0,3).contiguous()
            dec_output = self.permute(dec_output, (2, 1, 0, 3))
            time_embed_expanded = self.repeat(self.time_embed, (batch_size, 1, 1))
            dec_output = dec_output + time_embed_expanded
            # dec_output=dec_output.permute(2,1,0,3).contiguous()
            dec_output = self.permute(dec_output, (2, 1, 0, 3))
            
            

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask