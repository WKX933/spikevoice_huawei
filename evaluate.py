import argparse
import os

import mindspore
import yaml
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context, Tensor
from mindspore.train.summary import SummaryRecord

from utils.model import get_model, get_vocoder
from utils.tools import log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

# 设置MindSpore运行环境
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")  # 根据实际情况选择GPU或CPU

def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    
    # 使用MindSpore的GeneratorDataset
    loader = ds.GeneratorDataset(
        dataset,
        column_names=dataset.get_column_names(),
        shuffle=False,
        num_parallel_workers=4
    )
    loader = loader.batch(batch_size=batch_size, drop_remainder=False)
    
    # 添加后处理操作
    loader = loader.map(
        operations=dataset.postprocess,
        input_columns=dataset.get_column_names(),
        output_columns=["ids", "raw_texts", "speakers", "texts", "text_lens", 
                       "max_text_len", "mels", "mel_lens", "max_mel_len", 
                       "pitches", "energies", "durations"]
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    total_samples = 0
    
    # 设置模型为评估模式
    model.set_train(False)
    
    for batch in loader.create_dict_iterator():
        # MindSpore数据已经是Tensor格式，不需要手动转移到设备
        # 但需要确保数据类型正确
        texts = Tensor(batch["texts"], mindspore.float32)
        text_lens = Tensor(batch["text_lens"], mindspore.int32)
        mels = Tensor(batch["mels"], mindspore.float32)
        mel_lens = Tensor(batch["mel_lens"], mindspore.int32)
        max_mel_len = batch["max_mel_len"]
        pitches = Tensor(batch["pitches"], mindspore.float32)
        energies = Tensor(batch["energies"], mindspore.float32)
        durations = Tensor(batch["durations"], mindspore.int32)
        
        batch_size_current = len(batch["ids"])
        total_samples += batch_size_current
        
        # 使用MindSpore的no_grad等价方式
        mindspore.ops.StopGradient(True)
        # Forward
        output = model(texts, text_lens, mels, mel_lens, max_mel_len, pitches, energies, durations)

        # Cal Loss
        losses = Loss(batch, output)

        for i in range(len(losses)):
            # MindSpore的loss可能是Tensor，需要转换为numpy数值
            if hasattr(losses[i], 'asnumpy'):
                loss_value = losses[i].asnumpy()
            else:
                loss_value = losses[i]
            loss_sums[i] += loss_value * batch_size_current
        mindspore.ops.StopGradient(False)

    loss_means = [loss_sum / total_samples for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        # 准备用于合成的batch数据
        synth_batch = {
            "ids": batch["ids"],
            "raw_texts": batch["raw_texts"],
            "speakers": batch["speakers"],
            "texts": texts,
            "text_lens": text_lens,
            "mels": mels,
            "mel_lens": mel_lens,
            "pitches": pitches,
            "energies": energies,
            "durations": durations
        }
        
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            synth_batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    # 恢复训练模式（如果需要）
    model.set_train(True)
    
    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model - 移除device参数
    model = get_model(args, configs, train=False)

    message = evaluate(model, args.restore_step, configs)
    print(message)