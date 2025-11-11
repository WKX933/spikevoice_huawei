import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class FastSpeech2Loss(nn.Cell):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # 定义操作
        self.log = ops.Log()
        self.masked_select = ops.MaskedSelect()
        self.unsqueeze = ops.ExpandDims()
        self.logical_not = ops.LogicalNot()

    def construct(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        
        # 反转掩码
        src_masks = self.logical_not(src_masks)
        mel_masks = self.logical_not(mel_masks)
        
        # 计算对数持续时间目标
        log_duration_targets = self.log(duration_targets.astype(mindspore.float32) + 1)
        
        # 截断目标以匹配掩码形状
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        # MindSpore中不需要手动设置requires_grad

        # 根据特征级别选择pitch预测和目标
        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = self.masked_select(pitch_predictions, src_masks)
            pitch_targets = self.masked_select(pitch_targets, src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = self.masked_select(pitch_predictions, mel_masks)
            pitch_targets = self.masked_select(pitch_targets, mel_masks)

        # 根据特征级别选择energy预测和目标
        if self.energy_feature_level == "phoneme_level":
            energy_predictions = self.masked_select(energy_predictions, src_masks)
            energy_targets = self.masked_select(energy_targets, src_masks)
        elif self.energy_feature_level == "frame_level":
            energy_predictions = self.masked_select(energy_predictions, mel_masks)
            energy_targets = self.masked_select(energy_targets, mel_masks)

        # 选择持续时间预测和目标
        log_duration_predictions = self.masked_select(log_duration_predictions, src_masks)
        log_duration_targets = self.masked_select(log_duration_targets, src_masks)

        # 扩展mel掩码维度以匹配mel目标的维度
        mel_masks_expanded = self.unsqueeze(mel_masks, -1)
        
        # 选择mel预测和目标
        mel_predictions = self.masked_select(mel_predictions, mel_masks_expanded)
        postnet_mel_predictions = self.masked_select(
            postnet_mel_predictions, mel_masks_expanded
        )
        mel_targets = self.masked_select(mel_targets, mel_masks_expanded)

        # 计算各种损失
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # 计算总损失
        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )