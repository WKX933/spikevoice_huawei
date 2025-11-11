import mindspore
import numpy as np
import mindspore.nn as nn
from mindspore import ops


# class ScheduledOptim:
#     """ A simple wrapper class for learning rate scheduling """

#     def __init__(self, model, train_config, model_config, current_step):
#         # MindSpore使用不同的优化器API
#         self._optimizer = nn.Adam(
#             model.trainable_params(),
#             learning_rate=1.0,  # 初始学习率，会在_update_learning_rate中更新
#             beta1=train_config["optimizer"]["betas"][0],
#             beta2=train_config["optimizer"]["betas"][1],
#             eps=train_config["optimizer"]["eps"],
#             weight_decay=train_config["optimizer"]["weight_decay"],
#         )
#         self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
#         self.anneal_steps = train_config["optimizer"]["anneal_steps"]
#         self.anneal_rate = train_config["optimizer"]["anneal_rate"]
#         self.current_step = current_step
#         self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)
        
#         # 保存模型参数用于更新学习率
#         self.model = model

#     def step_and_update_lr(self):
#         self._update_learning_rate()
#         # MindSpore中优化器步骤需要梯度作为参数
#         # 在实际训练循环中，我们会在计算梯度后调用这个函数

#     def get_optimizer(self):
#         """返回优化器实例"""
#         return self._optimizer

#     def zero_grad(self):
#         """清空梯度"""
#         self._optimizer.zero_grad()

#     def load_state_dict(self, state_dict):
#         """加载状态字典"""
#         # MindSpore的优化器状态加载方式不同
#         # 需要在外部通过mindspore.load_param_into_net处理
#         pass

#     def _get_lr_scale(self):
#         lr = np.min(
#             [
#                 np.power(self.current_step, -0.5),
#                 np.power(self.n_warmup_steps, -1.5) * self.current_step,
#             ]
#         )
#         for s in self.anneal_steps:
#             if self.current_step > s:
#                 lr = lr * self.anneal_rate
#         return lr

#     def _update_learning_rate(self):
#         """ Learning rate scheduling per step """
#         self.current_step += 1
#         lr = self.init_lr * self._get_lr_scale()

#         # MindSpore中更新学习率的方式
#         # 需要重新创建优化器或者使用动态学习率
#         self._update_optimizer_lr(lr)

#     def _update_optimizer_lr(self, lr):
#         """更新优化器的学习率"""
#         # MindSpore中可以通过重新创建优化器来更新学习率
#         # 或者使用动态学习率API
#         train_config = {
#             "betas": (self._optimizer.beta1, self._optimizer.beta2),
#             "eps": self._optimizer.eps,
#             "weight_decay": self._optimizer.weight_decay
#         }
        
#         # 重新创建优化器 with new learning rate
#         self._optimizer = nn.Adam(
#             self.model.trainable_params(),
#             learning_rate=lr,
#             beta1=train_config["betas"][0],
#             beta2=train_config["betas"][1],
#             eps=train_config["eps"],
#             weight_decay=train_config["weight_decay"],
#         )

#     def state_dict(self):
#         """返回优化器状态字典"""
#         # 在MindSpore中，优化器状态管理方式不同
#         # 返回一个空字典，实际状态管理在外部处理
#         return {}

#     def get_current_lr(self):
#         """获取当前学习率"""
#         return self.init_lr * self._get_lr_scale()


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, model, train_config, model_config, current_step):
        self._optimizer = nn.Adam(
            model.trainable_params(),
            learning_rate=1.0,  # 初始学习率，会在_update_learning_rate中更新
            beta1=train_config["optimizer"]["betas"][0],
            beta2=train_config["optimizer"]["betas"][1],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)
        self.model = model

    # ✅ 关键：让该类可被直接调用
    def __call__(self, grads):
        """接受梯度并执行一步优化 + 学习率更新"""
        # 更新学习率
        self._update_learning_rate()

        # 梯度更新
        self._optimizer(grads)

        # 自增步数
        self.current_step += 1

    def step_and_update_lr(self):
        """备用接口（可直接调用）"""
        self._update_learning_rate()

    def get_optimizer(self):
        return self._optimizer

    def zero_grad(self):
        """MindSpore中通常不需要显式清零梯度，但保留接口"""
        pass

    def load_state_dict(self, state_dict):
        pass

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """更新学习率"""
        lr = self.init_lr * self._get_lr_scale()

        # 用 MindSpore 动态学习率 API 更新
        for p in self._optimizer.parameters:
            if hasattr(p, "learning_rate"):
                p.learning_rate = lr

        # 兼容旧写法（直接替换 optimizer）
        # breakpoint()
        # self._optimizer = nn.Adam(self.model.trainable_params(),learning_rate=lr,beta1=self._optimizer.beta1,beta2=self._optimizer.beta2,eps=self._optimizer.eps,weight_decay=float(self._optimizer.weight_decay.asnumpy()))
        self._optimizer = nn.Adam(self.model.trainable_params(),learning_rate=lr,beta1=float(self._optimizer.beta1.asnumpy()), eps=float(self._optimizer.eps.asnumpy()),weight_decay=float(self._optimizer.weight_decay.asnumpy()) )

    def state_dict(self):
        return {}

    def get_current_lr(self):
        """获取当前学习率"""
        return self.init_lr * self._get_lr_scale()


# 替代方案：使用MindSpore的动态学习率
class WarmupAnnealingLR:
    """MindSpore版本的预热和退火学习率调度器"""
    
    def __init__(self, base_lr, n_warmup_steps, anneal_steps, anneal_rate, current_step=0):
        self.base_lr = base_lr
        self.n_warmup_steps = n_warmup_steps
        self.anneal_steps = anneal_steps
        self.anneal_rate = anneal_rate
        self.current_step = current_step
        
    def get_lr(self):
        """获取当前步的学习率"""
        lr_scale = np.min([
            np.power(self.current_step, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.current_step,
        ])
        
        lr = self.base_lr * lr_scale
        
        for step in self.anneal_steps:
            if self.current_step > step:
                lr = lr * self.anneal_rate
                
        self.current_step += 1
        return lr
