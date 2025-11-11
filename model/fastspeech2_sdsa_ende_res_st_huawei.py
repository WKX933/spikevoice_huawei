import os
import json

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

import mindspore as ms

from transformer_sdsa_ende_res_st_hw import Encoder, Decoder, PostNet
from .modules_attn_res_huawei import VarianceAdaptor
from utils.tools import get_mask_from_lengths

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


class FastSpeech2_sdsa_ende_res_st(nn.Cell):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2_sdsa_ende_res_st, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Dense(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()
        self.lif_m = MultiStepLIFNode()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        
        # 定义操作
        self.unsqueeze = ops.ExpandDims()
        self.expand = ops.BroadcastTo
        self.mean = ops.ReduceMean()
        self.shape = ops.Shape()

    def construct(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        # breakpoint()
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        # (12,124) (12,124)
        # breakpoint()
        output = self.encoder(texts, src_masks)
        # (4,24,124,256)
        # output = output.mean(0)

        if self.speaker_emb is not None:
            speaker_emb = self.speaker_emb(speakers)
            speaker_emb_expanded = self.unsqueeze(speaker_emb, 1)
            # 扩展维度以匹配output
            T, batch_size, seq_len, hidden_dim = self.shape(output)
            speaker_emb_expanded = self.expand((T, batch_size, seq_len, hidden_dim))(
                speaker_emb_expanded
            )
            output = output + speaker_emb_expanded
        # breakpoint()
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        # breakpoint()
        output, mel_masks = self.decoder(output, mel_masks)
        # breakpoint()
        # (24,860,256)
        output = self.lif_m(output)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        # 对时间维度求平均
        output_mean = self.mean(output, 0)
        postnet_output_mean = self.mean(postnet_output, 0)

        # return output_mean

        return (
            output_mean,
            postnet_output_mean,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks[0],
            src_lens,
            mel_lens,
        )


# if __name__ == "__main__":
#     import numpy as np
#     from mindspore import context
#     import yaml

#     context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

#     # ====== 模拟配置 ======
#     preprocess_config = {
#         "path": {"preprocessed_path": "./"},
#         "preprocessing": {"mel": {"n_mel_channels": 80}},
#     }

#     model_config = {
#         "transformer": {
#             "encoder_hidden": 256,
#             "decoder_hidden": 256,
#         },
#         "multi_speaker": False,
#         "max_seq_len": 200,
#     }

#     # ====== 构造模型 ======
#     model = FastSpeech2_sdsa_ende_res_st(preprocess_config, model_config)

#     # ====== 伪造输入 ======
#     batch_size = 2
#     seq_len = 100
#     speakers = mindspore.Tensor(np.zeros((batch_size,), dtype=np.int32))
#     texts = mindspore.Tensor(np.random.randint(0, 100, (batch_size, seq_len)), mindspore.int32)
#     src_lens = mindspore.Tensor(np.array([seq_len, seq_len]), mindspore.int32)
#     max_src_len = seq_len

#     # ====== 前向推理 ======
#     outputs = model(speakers, texts, src_lens, max_src_len)
#     print("✅ Forward pass success!")
#     print("Output shapes:")
#     for o in outputs[:2]:
#         print(o.shape)


# if __name__ == "__main__":
#     import yaml
#     import numpy as np
#     from mindspore import context, Tensor
#     import mindspore as ms

#     context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

#     # ====== 固定路径加载配置 ======
#     model_config_path = "/home/mseco/spikevoice/SpikeVoice_hw/config/LJSpeech/model.yaml"

#     with open(model_config_path, "r") as f:
#         model_config = yaml.load(f, Loader=yaml.FullLoader)

#     # 构造最小 preprocess_config（可根据你真实路径调整）
#     preprocess_config = {
#         "path": {"preprocessed_path": "./"},
#         "preprocessing": {"mel": {"n_mel_channels": 80}},
#     }

#     # ====== 构造模型 ======
#     model = FastSpeech2_sdsa_ende_res_st(preprocess_config, model_config)

#     # ====== 伪造输入进行一次前向推理 ======
#     batch_size = 2
#     seq_len = 100
#     speakers = Tensor(np.zeros((batch_size,), dtype=np.int32))
#     texts = Tensor(np.random.randint(0, 100, (batch_size, seq_len)), ms.int32)
#     src_lens = Tensor(np.array([seq_len, seq_len]), ms.int32)
#     max_src_len = seq_len

#     outputs = model(speakers, texts, src_lens, max_src_len)
#     print("✅ Forward pass success!")
#     print("Output shapes:")
#     for o in outputs[:2]:
#         print(o.shape)
