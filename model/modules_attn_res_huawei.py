import os
import json
import copy
import math
from collections import OrderedDict

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

from mindspore import Tensor, dtype as mstype


from utils.tools import get_mask_from_lengths, pad
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
        grad_input = ops.where(x < 0, mindspore.Tensor(0.0, mindspore.float32), grad_input)
        grad_input = ops.where(x > 4, mindspore.Tensor(0.0, mindspore.float32), grad_input)
        return (grad_input,)


class MultiStepLIFNode(nn.Cell):
    """
    Multi-step Leaky Integrate-and-Fire node
    with quantized spike activation.
    """

    def __init__(self, decay=0.2, act=False):
        super(MultiStepLIFNode, self).__init__()
        self.decay = mindspore.Tensor(decay, mindspore.float32)
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


# class VarianceAdaptor(nn.Cell):
#     """Variance Adaptor"""

#     def __init__(self, preprocess_config, model_config):
#         super(VarianceAdaptor, self).__init__()
#         self.duration_predictor = VariancePredictor(model_config)
#         self.length_regulator = LengthRegulator()
#         self.pitch_predictor = VariancePredictor(model_config)
#         self.energy_predictor = VariancePredictor(model_config)
#         self.lif_pitch = MultiStepLIFNode()
#         self.lif_energy = MultiStepLIFNode()

#         self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
#             "feature"
#         ]
#         self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
#             "feature"
#         ]
#         assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
#         assert self.energy_feature_level in ["phoneme_level", "frame_level"]

#         self.T = model_config["T"]

#         pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
#         energy_quantization = model_config["variance_embedding"]["energy_quantization"]
#         n_bins = model_config["variance_embedding"]["n_bins"]
#         assert pitch_quantization in ["linear", "log"]
#         assert energy_quantization in ["linear", "log"]
#         with open(
#             os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
#         ) as f:
#             stats = json.load(f)
#             pitch_min, pitch_max = stats["pitch"][:2]
#             energy_min, energy_max = stats["energy"][:2]

#         if pitch_quantization == "log":
#             self.pitch_bins = mindspore.Parameter(
#                 ops.Exp()(ops.LinSpace()(np.log(pitch_min), np.log(pitch_max), n_bins - 1)),
#                 requires_grad=False,
#             )
#         else:
#             self.pitch_bins = mindspore.Parameter(
#                 # ops.LinSpace()(pitch_min, pitch_max, n_bins - 1),
#                 # requires_grad=False,
#                 ops.LinSpace()(
#                     Tensor(pitch_min, mstype.float32),
#                     Tensor(pitch_max, mstype.float32),
#                     Tensor(n_bins - 1, mstype.int32)
#                 ),
#                 requires_grad=False,
#             )
#         if energy_quantization == "log":
#             self.energy_bins = mindspore.Parameter(
#                 ops.Exp()(ops.LinSpace()(np.log(energy_min), np.log(energy_max), n_bins - 1)),
#                 requires_grad=False,
#             )
#         else:
#             self.energy_bins = mindspore.Parameter(
#                 ops.LinSpace()(
#                     Tensor(energy_min, mstype.float32),
#                     Tensor(energy_max, mstype.float32),
#                     Tensor(n_bins - 1, mstype.int32)
#                 ),
#                 requires_grad=False,
#             )

#         self.pitch_embedding = nn.Embedding(
#             n_bins, model_config["transformer"]["encoder_hidden"]
#         )
#         self.energy_embedding = nn.Embedding(
#             n_bins, model_config["transformer"]["encoder_hidden"]
#         )
        
#         # 定义操作
#         self.unsqueeze = ops.ExpandDims()
#         self.repeat = ops.Tile()
#         self.clamp = ops.clip_by_value
#         self.round = ops.Round()
#         self.exp = ops.Exp()
#         self.mean = ops.ReduceMean()
#         self.bucketize = ops.Bucketize(boundaries=self.pitch_bins.asnumpy().tolist())

#     def get_pitch_embedding(self, x, target, mask, control):
#         prediction = self.pitch_predictor(x, mask)
#         # (24,119)
#         if target is not None:
#             # 创建针对pitch的bucketize操作
#             pitch_bucketize = ops.Bucketize(boundaries=self.pitch_bins)
#             embedding = self.pitch_embedding(pitch_bucketize(target))
#         else:
#             prediction = prediction * control
#             pitch_bucketize = ops.Bucketize(boundaries=self.pitch_bins)
#             embedding = self.pitch_embedding(
#                 pitch_bucketize(prediction)
#             )
#         # embedding = self.atan_pitch(embedding)
#         # embedding(24,119,256)
#         return prediction, embedding

#     def get_energy_embedding(self, x, target, mask, control):
#         prediction = self.energy_predictor(x, mask)
#         if target is not None:
#             # 创建针对energy的bucketize操作
#             energy_bucketize = ops.Bucketize(boundaries=self.energy_bins)
#             embedding = self.energy_embedding(energy_bucketize(target))
#         else:
#             prediction = prediction * control
#             energy_bucketize = ops.Bucketize(boundaries=self.energy_bins)
#             embedding = self.energy_embedding(
#                 energy_bucketize(prediction)
#             )
#         # embedding = self.atan_energy(embedding)
#         return prediction, embedding

#     def construct(
#         self,
#         x,
#         src_mask,
#         mel_mask=None,
#         max_len=None,
#         pitch_target=None,
#         energy_target=None,
#         duration_target=None,
#         p_control=1.0,
#         e_control=1.0,
#         d_control=1.0,
#     ):
#         # x(24,116,256) pitch_target(24,112) energy_target(24,112)
#         if self.training:
#             pitch_target = self.repeat(self.unsqueeze(pitch_target, 0), (self.T, 1, 1))
#             # (4,24,112)
#             energy_target = self.repeat(self.unsqueeze(energy_target, 0), (self.T, 1, 1))
#             # (4,24,112)

#         log_duration_prediction = self.duration_predictor(x, src_mask)
#         # not spike
#         # (24,116)
#         if self.pitch_feature_level == "phoneme_level":
#             # pitch_target(24,119),
#             pitch_prediction, pitch_embedding = self.get_pitch_embedding(
#                 x, pitch_target, src_mask, p_control
#             )
#             # prediction(24,119) embedding(24,119,256)
#             x = x + pitch_embedding
#         if self.energy_feature_level == "phoneme_level":
#             energy_prediction, energy_embedding = self.get_energy_embedding(
#                 x, energy_target, src_mask, p_control
#             )
#             x = x + energy_embedding

#         if duration_target is not None:
#             # duration_target(24,123)
#             x, mel_len = self.length_regulator(x, duration_target, max_len)
#             # x(24,864,256) mel_len(24)
#             duration_rounded = duration_target
#         else:
#             duration_rounded = self.clamp(
#                 (self.round(self.exp(log_duration_prediction.mean(0)) - 1) * d_control),
#                 clip_value_min=0,
#             )
#             x, mel_len = self.length_regulator(x, duration_rounded, max_len)
#             mel_mask = get_mask_from_lengths(mel_len)

#         if self.pitch_feature_level == "frame_level":
#             pitch_prediction, pitch_embedding = self.get_pitch_embedding(
#                 x, pitch_target, mel_mask, p_control
#             )
#             x = x + pitch_embedding
#         if self.energy_feature_level == "frame_level":
#             energy_prediction, energy_embedding = self.get_energy_embedding(
#                 x, energy_target, mel_mask, p_control
#             )
#             x = x + energy_embedding

#         return (
#             x,
#             pitch_prediction.mean(0),
#             energy_prediction.mean(0),
#             log_duration_prediction.mean(0),
#             duration_rounded,
#             mel_len,
#             mel_mask,
#         )


class VarianceAdaptor(nn.Cell):
    """MindSpore 版本 VarianceAdaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.lif_pitch = MultiStepLIFNode()
        self.lif_energy = MultiStepLIFNode()

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        self.T = model_config["T"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        # ✅ 改成 Python list，不再是 Parameter
        if pitch_quantization == "log":
            self.pitch_bins = np.exp(
                np.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
            ).tolist()
        else:
            self.pitch_bins = np.linspace(pitch_min, pitch_max, n_bins - 1).tolist()

        if energy_quantization == "log":
            self.energy_bins = np.exp(
                np.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
            ).tolist()
        else:
            self.energy_bins = np.linspace(energy_min, energy_max, n_bins - 1).tolist()

        self.pitch_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])
        self.energy_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])

        # ops
        self.unsqueeze = ops.ExpandDims()
        self.repeat = ops.Tile()
        self.clamp = ops.clip_by_value
        self.round = ops.Round()
        self.exp = ops.Exp()
        self.mean = ops.ReduceMean()

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        pitch_bucketize = ops.Bucketize(boundaries=self.pitch_bins)
        if target is not None:
            embedding = self.pitch_embedding(pitch_bucketize(target))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(pitch_bucketize(prediction))
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        energy_bucketize = ops.Bucketize(boundaries=self.energy_bins)
        if target is not None:
            embedding = self.energy_embedding(energy_bucketize(target))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(energy_bucketize(prediction))
        return prediction, embedding

    def construct(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        if self.training:
            pitch_target = self.repeat(self.unsqueeze(pitch_target, 0), (self.T, 1, 1))
            energy_target = self.repeat(self.unsqueeze(energy_target, 0), (self.T, 1, 1))

        log_duration_prediction = self.duration_predictor(x, src_mask)

        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, src_mask, e_control)
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = self.clamp(
                (self.round(self.exp(log_duration_prediction.mean(0)) - 1) * d_control),
                clip_value_min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mel_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mel_mask, e_control)
            x = x + energy_embedding

        return (
            x,
            pitch_prediction.mean(0),
            energy_prediction.mean(0),
            log_duration_prediction.mean(0),
            duration_rounded,
            mel_len,
            mel_mask,
        )

class LengthRegulator(nn.Cell):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()
        # self.repeat_interleave = ops.RepeatInterleave()
        self.stack = ops.Stack()
        self.tile = ops.Tile()

    def LR(self, x, duration, max_len):
        """
        x: (T, B, L, H)
        duration: (B, L)
        """
        output = []
        mel_len = []

        T, batch_size, seq_len, hidden_dim = x.shape

        for i in range(T):
            out = []
            for j in range(batch_size):
                frame_pieces = []
                for k in range(seq_len):
                    repeat_count = int(duration[j][k].asnumpy())  # ✅ 转为Python int
                    if repeat_count > 0:
                        expanded = ops.tile(x[i, j, k].expand_dims(0), (repeat_count, 1))
                        frame_pieces.append(expanded)
                if frame_pieces:
                    concat = ops.concat(frame_pieces, axis=0)
                    out.append(concat)
                    if i == 0:
                        mel_len.append(concat.shape[0])

            if out:
                # padding 对齐
                max_t = max([t.shape[0] for t in out]) if max_len is None else max_len
                out_padded = []
                for o in out:
                    pad_len = max_t - o.shape[0]
                    if pad_len > 0:
                        pad_tensor = ops.zeros((pad_len, hidden_dim), x.dtype)
                        out_padded.append(ops.concat((o, pad_tensor), axis=0))
                    else:
                        out_padded.append(o)
                output.append(self.stack(out_padded))

        output = self.stack(output)
        mel_len = Tensor(mel_len, mindspore.int32)
        return output, mel_len
        
    def expand(self, batch, predicted):
        out = list()

        for j, batch_t in enumerate(batch):
            out.append(self.repeat_interleave(batch_t, predicted, axis=0))

        return out

    def construct(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


# class LengthRegulator(nn.Cell):
#     def __init__(self):
#         super(LengthRegulator, self).__init__()
#         self.stack = ops.Stack()
#         self.tile = ops.Tile()

#     def LR(self, x, duration, max_len):
#         """
#         x: (T, B, L, H)
#         duration: (B, L)
#         """
#         output = []
#         mel_len = []

#         T, batch_size, seq_len, hidden_dim = x.shape

#         for i in range(T):
#             out = []
#             for j in range(batch_size):
#                 frame_pieces = []
#                 for k in range(seq_len):
#                     repeat_count = int(duration[j][k].asnumpy())  # ✅ 转为Python int
#                     if repeat_count > 0:
#                         expanded = ops.tile(x[i, j, k].expand_dims(0), (repeat_count, 1))
#                         frame_pieces.append(expanded)
#                 if frame_pieces:
#                     concat = ops.concat(frame_pieces, axis=0)
#                     out.append(concat)
#                     if i == 0:
#                         mel_len.append(concat.shape[0])

#             if out:
#                 # padding 对齐
#                 max_t = max([t.shape[0] for t in out]) if max_len is None else max_len
#                 out_padded = []
#                 for o in out:
#                     pad_len = max_t - o.shape[0]
#                     if pad_len > 0:
#                         pad_tensor = ops.zeros((pad_len, hidden_dim), x.dtype)
#                         out_padded.append(ops.concat((o, pad_tensor), axis=0))
#                     else:
#                         out_padded.append(o)
#                 output.append(self.stack(out_padded))

#         output = self.stack(output)
#         mel_len = Tensor(mel_len, mindspore.int32)
#         return output, mel_len


class VariancePredictor(nn.Cell):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv1d_1 = Conv(
            self.input_size,
            self.filter_size,
            kernel_size=self.kernel,
            padding=(self.kernel - 1) // 2,
        )
        self.lif_1 = MultiStepLIFNode()
        self.layer_norm_1 = nn.LayerNorm([self.filter_size])
        self.dropout_1 = nn.Dropout(p=self.dropout)

        self.conv1d_2 = Conv(
            self.filter_size,
            self.filter_size,
            kernel_size=self.kernel,
            padding=1,
        )
        self.lif_2 = MultiStepLIFNode()
        self.layer_norm_2 = nn.LayerNorm([self.filter_size])
        self.dropout_2 = nn.Dropout(p=self.dropout)
        self.linear_layer = nn.Dense(self.conv_output_size, 1)
        self.lif_3 = MultiStepLIFNode()
        self.lif_0 = MultiStepLIFNode()
        
        # 定义操作
        self.reshape = ops.Reshape()
        self.squeeze = ops.Squeeze(-1)
        self.masked_fill = ops.MaskedFill()
        self.shape = ops.Shape()

    def construct(self, encoder_output, mask):
        # (4,24,116,256)
        T, B, L, _ = self.shape(encoder_output)
        encoder_output = self.lif_0(encoder_output)
        encoder_output = self.reshape(encoder_output, (T * B, L, -1))
        encoder_output = self.reshape(self.conv1d_1(encoder_output), (T, B, L, -1))
        encoder_output = self.lif_1(encoder_output)
        encoder_output = self.layer_norm_1(encoder_output)
        # encoder_output = self.dropout_1(encoder_output)

        encoder_output = self.reshape(encoder_output, (T * B, L, -1))
        encoder_output = self.reshape(self.conv1d_2(encoder_output), (T, B, L, -1))
        encoder_output = self.lif_2(encoder_output)
        encoder_output = self.layer_norm_2(encoder_output)
        # out = self.dropout_2(encoder_output)
        # out = self.conv_layer(encoder_output)
        # (24,116,256)
        out = self.linear_layer(encoder_output)
        # out = self.lif_3(out)
        # (24,116,1)
        out = self.squeeze(out)

        if mask is not None:
            out = self.masked_fill(out, mask, 0.0)

        return out


class Conv(nn.Cell):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

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
        
        self.transpose = ops.Transpose()

    def construct(self, x):
        x = self.transpose(x, (0, 2, 1))
        x = self.conv(x)
        x = self.transpose(x, (0, 2, 1))

        return x