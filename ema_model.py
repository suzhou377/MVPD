import torch
class EMAModel(torch.nn.Module):
    def __init__(self, model, ema_model, update_bn=True):
        super(EMAModel, self).__init__()  # 调用基类的构造函数
        self.model = model
        self.ema_model = ema_model
        self.update_bn = update_bn
        self.decay_rate = 0.
    def forward(self, x):
        x = self.ema_model(x)
        return x
    def update(self, epoch, ema_epoch, decay):
        if epoch < ema_epoch:
            self.decay_rate = 0.
        else:
            self.decay_rate = decay
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.decay_rate).add_(param.data, alpha=1 - self.decay_rate)
            if self.update_bn:
                for module, ema_module in zip(self.model.modules(), self.ema_model.modules()):
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        ema_module.running_mean.mul_(self.decay_rate).add_(module.running_mean,
                                                                           alpha=1 - self.decay_rate)
                        ema_module.running_var.mul_(self.decay_rate).add_(module.running_var, alpha=1 - self.decay_rate)
                        ema_module.num_batches_tracked = module.num_batches_tracked

