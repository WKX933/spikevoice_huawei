import os
import json

import mindspore
import numpy as np
import mindspore.ops as ops

import hifigan
from model.optimizer_huawei import ScheduledOptim
from model.fastspeech2_sdsa_ende_res_st_huawei import FastSpeech2_sdsa_ende_res_st

def get_model(args, configs, train=False, mode='baseline'):
    (preprocess_config, model_config, train_config) = configs
    # if mode == 'baseline':
    #     model = FastSpeech2(preprocess_config, model_config)
    if mode == 'sdsa_ende_res_st':
        model = FastSpeech2_sdsa_ende_res_st(preprocess_config, model_config)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.ckpt".format(args.restore_step),  # 改为MindSpore的ckpt格式
        )
        # 加载MindSpore checkpoint
        param_dict = mindspore.load_checkpoint(ckpt_path)
        mindspore.load_param_into_net(model, param_dict)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            # 在MindSpore中优化器状态需要单独处理
            pass
        model.set_train(True)
        return model, scheduled_optim

    model.set_train(False)
    # MindSpore中不需要设置requires_grad，通过set_train控制
    return model

def get_model_attn(args, configs, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2_attn(preprocess_config, model_config)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.ckpt".format(args.restore_step),
        )
        param_dict = mindspore.load_checkpoint(ckpt_path)
        mindspore.load_param_into_net(model, param_dict)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            # 优化器状态处理
            pass
        model.set_train(True)
        return model, scheduled_optim

    model.set_train(False)
    return model


def get_param_num(model):
    num_param = 0
    for param in model.get_parameters():
        num_param += param.size
    return num_param


def get_vocoder(config):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        # MindSpore目前没有直接的torch.hub等价物，需要手动加载
        if speaker == "LJSpeech":
            # 需要适配MindSpore的MelGAN实现
            vocoder = None  # 替换为MindSpore兼容的MelGAN
        elif speaker == "universal":
            vocoder = None  # 替换为MindSpore兼容的MelGAN
        # vocoder.mel2wav.set_train(False)
        # vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config_data = json.load(f)
        config = hifigan.AttrDict(config_data)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            # 加载MindSpore格式的checkpoint
            ckpt_path = "hifigan/generator_LJSpeech.ckpt"
        elif speaker == "universal":
            ckpt_path = "hifigan/generator_universal.ckpt"
        
        # 加载参数
        # param_dict = mindspore.load_checkpoint(ckpt_path)
        # mindspore.load_param_into_net(vocoder, param_dict)
        
        vocoder.set_train(False)
        # MindSpore中移除权重归一化的方式可能不同
        # vocoder.remove_weight_norm()

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    
    # 使用MindSpore的no_grad等价方式
    mindspore.ops.StopGradient(True)
    
    if name == "MelGAN":
        # MindSpore版本的MelGAN推理
        wavs = vocoder.inverse(mels / np.log(10))
    elif name == "HiFi-GAN":
        wavs = vocoder(mels)
        # 压缩维度
        wavs = ops.Squeeze(1)(wavs)
    
    mindspore.ops.StopGradient(False)

    # 转换为numpy并处理
    wavs = (
        wavs.asnumpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs