import argparse
import os
# os.chdir('/opt/tiger/transformer-tts/FastSpeech2')
os.environ["DEVICE_ID"] = "0"  
import mindspore
import yaml
import mindspore.nn as nn
from mindspore import context,ops
from mindspore.dataset import GeneratorDataset  
from mindspore.train.summary import SummaryRecord  
from tqdm import tqdm
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
# from spikingjelly.clock_driven import functional
from evaluate import evaluate
import numpy as np
import random
from mindspore.ops import composite as C

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")  # Ascend

def setup_seed(seed):
    mindspore.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # MindSpore没有直接对应的cudnn.deterministic

# 设置随机数种子
setup_seed(20)

grad = C.GradOperation(get_by_list=True)

def main(args, configs):
    print("Prepare training ...")
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 1
    # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    
    # MindSpore的数据加载方式
    loader = GeneratorDataset(
        dataset, 
        column_names=dataset.get_column_names(),  # 需要数据集提供列名
        shuffle=True,
        num_parallel_workers=4
    )
    loader = loader.batch(batch_size=batch_size * group_size, drop_remainder=True)

    loader = loader.map(
        operations=dataset.postprocess,
        input_columns=dataset.get_column_names(),
        output_columns=["ids", "raw_texts", "speakers", "texts", "text_lens", 
                       "max_text_len", "mels", "mel_lens", "max_mel_len", 
                       "pitches", "energies", "durations"]
    )

    # Prepare model
    model, optimizer = get_model(args, configs, train=True, mode='sdsa_ende_res_st')
    # breakpoint()
    # MindSpore没有DataParallel，使用Model类或自定义并行
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    
    # MindSpore的日志记录器
    train_logger = SummaryRecord(train_log_path)
    # val_logger = SummaryRecord(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    # breakpoint()

    # 定义训练步骤
    def train_step(batch):
        # 前向计算
        output = model(*batch[2:])
        losses = Loss(batch, output)
        total_loss = losses[0]

        # 获取可训练参数
        weights = model.trainable_params()

        # 计算梯度
        grads = grad(model, weights)(*batch[2:])   # 自动传递输入
        grads = [g for g in grads if g is not None]

        # ✅ 扁平化梯度
        flat_grads = []
        for g in grads:
            if isinstance(g, (list, tuple)):
                flat_grads.extend(g)
            else:
                flat_grads.append(g)

        # ✅ 全局梯度裁剪
        grads = ops.clip_by_global_norm(flat_grads, grad_clip_thresh)

        # ✅ 更新参数
        optimizer(grads)
        return losses, total_loss

    while True:
        # SpikingJelly重置，可能需要调整
        # functional.reset_net(model)
        
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        
        for batch in loader:
            # breakpoint()
            # MindSpore不需要手动转移到设备，但需要确保数据格式正确
            batch = [mindspore.Tensor(b) for b in batch]
            # breakpoint()
            
            # 训练步骤
            losses, total_loss = train_step(batch)
            # breakpoint()
            
            if step % grad_acc_step == 0:
                # 在MindSpore中，优化器步骤已经在train_step中执行
                pass

            if step % log_step == 0:
                losses = [l.asnumpy().item() if hasattr(l, 'asnumpy') else l for l in losses]
                message1 = "Step {}/{}, ".format(step, total_step)
                message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                    *losses
                )
                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")
                outer_bar.write(message1 + message2)
                
                log(train_logger, step, losses=losses)

            if step == total_step:
                train_logger.close()
                # val_logger.close()
                return
                
            step += 1
            outer_bar.update(1)
            inner_bar.update(1)
            
        epoch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml",
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

    main(args, configs)