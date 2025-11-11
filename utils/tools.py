from cmath import inf
import os
import json

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
from mindspore import dtype as mstype


matplotlib.use("Agg")

def to_device(data):
    """
    MindSpore自动处理设备，此函数主要进行数据类型转换
    """
    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = mindspore.Tensor(speakers, dtype=mindspore.int64)
        texts = mindspore.Tensor(texts, dtype=mindspore.int64)
        src_lens = mindspore.Tensor(src_lens)
        mels = mindspore.Tensor(mels, dtype=mindspore.float32)
        mel_lens = mindspore.Tensor(mel_lens)
        pitches = mindspore.Tensor(pitches, dtype=mindspore.float32)
        energies = mindspore.Tensor(energies)
        durations = mindspore.Tensor(durations, dtype=mindspore.int64)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = mindspore.Tensor(speakers, dtype=mindspore.int64)
        texts = mindspore.Tensor(texts, dtype=mindspore.int64)
        src_lens = mindspore.Tensor(src_lens)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        breakpoint()
        logger.record("Loss/total_loss", losses[0], int(step))
        logger.record("Loss/mel_loss", losses[1], int(step))
        logger.record("Loss/mel_postnet_loss", losses[2], int(step))
        logger.record("Loss/pitch_loss", losses[3], int(step))
        logger.record("Loss/energy_loss", losses[4], int(step))
        logger.record("Loss/duration_loss", losses[5], int(step))

    if fig is not None:
        logger.record_figure(tag, fig)

    if audio is not None:
        # MindSpore的SummaryRecord可能需要不同的音频记录方式
        audio_normalized = audio / np.max(np.abs(audio))
        logger.record_audio(
            tag,
            audio_normalized,
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    # breakpoint()
    if max_len is None:
        max_len = int(ops.ReduceMax()(lengths).asnumpy())
    
    # try:
    #     max_len = int(ops.ReduceMax()(lengths).asnumpy())
    # except:
    #     breakpoint()

    # breakpoint()
    ids = ops.ExpandDims()(ops.arange(0, max_len), 0)
    ids = ops.Tile()(ids, (batch_size, 1))
    mask = ids >= ops.ExpandDims()(lengths, 1)

    return mask

# def get_mask_from_lengths(lengths, max_len=None):
#     batch_size = lengths.shape[0]
#     # breakpoint()
#     # if max_len.dtype == mstype.string:
#     max_len1 = int(ops.ReduceMax()(lengths).asnumpy())
    
#     try:
#         ids = ops.ExpandDims()(ops.arange(0, max_len1), 0)
#     except:
#         breakpoint()
#     ids = ops.Tile()(ids, (batch_size, 1))
#     mask = ids >= ops.ExpandDims()(lengths, 1)
#     return mask



# def get_mask_from_lengths(lengths, max_len=None):
#     # 确保 lengths 是 Tensor[int]
#     if not isinstance(lengths, ms.Tensor):
#         lengths = ms.Tensor(lengths, ms.int32)
#     elif lengths.dtype != ms.int32:
#         lengths = lengths.astype(ms.int32)

#     # 自动推导 max_len（忽略传入的字符串）
#     if (max_len is None) or (isinstance(max_len, ms.Tensor) and max_len.dtype == ms.string):
#         max_len = ops.max(lengths)
#     elif isinstance(max_len, ms.Tensor):
#         max_len = int(max_len.asnumpy().item())
#     elif isinstance(max_len, (list, tuple)):
#         max_len = int(max(max_len))
#     elif isinstance(max_len, str):
#         # 不能转换的字符串自动忽略
#         try:
#             max_len = int(max_len)
#         except:
#             max_len = int(ops.max(lengths).asnumpy())
#     else:
#         max_len = int(max_len)

#     # 构造 mask
#     ids = ops.arange(0, max_len, 1, ms.int32)
#     ids = ops.ExpandDims()(ids, 0)
#     ids = ops.Tile()(ids, (lengths.shape[0], 1))
#     mask = ids >= ops.ExpandDims()(lengths, 1)
#     return mask

# def get_mask_from_lengths(lengths, max_len=None):
#     lengths = mindspore.Tensor(lengths, mindspore.int32) if not isinstance(lengths, mindspore.Tensor) else lengths
#     batch_size = lengths.shape[0]
#     max_len = int(ops.max(lengths).asnumpy()) if max_len is None else int(max_len)
#     ids = ops.arange(0, mindspore.Tensor(max_len, mindspore.int32), 1, mindspore.int32)
#     ids = ops.ExpandDims()(ids, 0)
#     ids = ops.Tile()(ids, (batch_size, 1))
#     mask = ids >= ops.ExpandDims()(lengths, 1)
#     return mask



def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    src_len = int(predictions[8][0].asnumpy())
    mel_len = int(predictions[9][0].asnumpy())
    mel_target = ops.Transpose()(targets[6][0, :mel_len], (1, 0))
    mel_prediction = ops.Transpose()(predictions[1][0, :mel_len], (1, 0))
    duration = targets[11][0, :src_len].asnumpy()
    
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].asnumpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].asnumpy()
        
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].asnumpy()
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].asnumpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.asnumpy(), pitch, energy),
            (mel_target.asnumpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            ops.ExpandDims()(mel_target, 0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            ops.ExpandDims()(mel_prediction, 0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = int(predictions[8][i].asnumpy())
        mel_len = int(predictions[9][i].asnumpy())
        mel_prediction = ops.Transpose()(predictions[1][i, :mel_len], (1, 0))
        duration = predictions[5][i, :src_len].asnumpy()
        
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].asnumpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].asnumpy()
            
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].asnumpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].asnumpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.asnumpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = ops.Transpose()(predictions[1], (0, 2, 1))
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_1D_long(inputs, window, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    w = int(2 * window)
    len_pad = (int(max_len / w) + int(max_len % w > 0)) * w
    padded = np.stack([pad_data(x, len_pad, PAD) for x in inputs])

    return padded


def pad_2D_long(inputs, window, maxlen=None):
    w = window * 2
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        len_pad = (int(maxlen / w) + int(maxlen % w > 0)) * w
        output = np.stack([pad(x, len_pad) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        len_pad = (int(max_len / w) + int(max_len % w > 0)) * w
        output = np.stack([pad(x, len_pad) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].shape[0] for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            # 使用MindSpore的pad操作
            one_batch_padded = ops.Pad(((0, max_len - batch.shape[0])))(batch)
        elif len(batch.shape) == 2:
            one_batch_padded = ops.Pad(((0, max_len - batch.shape[0]), (0, 0)))(batch)
        out_list.append(one_batch_padded)
    
    out_padded = ops.Stack()(out_list)
    return out_padded


def mask_attn_unidir(attn_weights, mode='local'):
    if mode == 'local':
        win = attn_weights.shape[-1]
        l = int((win - 1) / 2)
        # 创建掩码
        mask = ops.ZerosLike()(attn_weights)
        # 偶数头掩码后l个位置
        mask[:, :, 0::2, -l:] = -float('inf')
        # 奇数头掩码前l个位置
        mask[:, :, 1::2, :l] = -float('inf')
        attn_weights = attn_weights + mask
        
    elif mode == 'global_l':
        bsz, length, heads, num_global = attn_weights.shape
        attn_weights = ops.Transpose()(attn_weights, (0, 2, 1, 3))
        
        # 创建上三角和下三角掩码
        for head_idx in range(heads):
            if head_idx % 2 == 0:  # 偶数头
                # 上三角掩码
                for i in range(length):
                    for j in range(i + 1, num_global):
                        if j < num_global:
                            attn_weights[:, head_idx, i, j] = -10000
            else:  # 奇数头
                # 下三角掩码
                for i in range(length):
                    for j in range(0, i):
                        if j < num_global:
                            attn_weights[:, head_idx, i, j] = -10000
                            
        attn_weights = ops.Transpose()(attn_weights, (0, 2, 1, 3))
        
    elif mode == 'global_r':
        bsz, heads, num_global, length = attn_weights.shape
        # 创建上三角和下三角掩码
        for head_idx in range(heads):
            if head_idx % 2 == 0:  # 偶数头
                # 上三角掩码
                for i in range(num_global):
                    for j in range(i + 1, length):
                        if j < length:
                            attn_weights[:, head_idx, i, j] = -10000
            else:  # 奇数头
                # 下三角掩码
                for i in range(num_global):
                    for j in range(0, i):
                        if j < length:
                            attn_weights[:, head_idx, i, j] = -10000

    return attn_weights