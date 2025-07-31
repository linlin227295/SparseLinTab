"""Utils for model/optimizer initialization and training."""
import pprint

from torch import optim

from sparselintab.utils.optim_utils import Lookahead, Lamb


def init_optimizer(c, model_parameters, device):
    if 'default' in c.exp_optimizer:
        optimizer = optim.Adam(params=model_parameters, lr=c.exp_lr)
    elif 'lamb' in c.exp_optimizer:  # 选择LAMB优化器
        lamb = Lamb
        optimizer = lamb(
            model_parameters, lr=c.exp_lr, betas=(0.9, 0.999),
            weight_decay=c.exp_weight_decay, eps=1e-6)
    else:
        raise NotImplementedError

    if c.exp_optimizer.startswith('lookahead_'):  # 检查配置中的优化器名称是否以lookahead_开头
        optimizer = Lookahead(optimizer, k=c.exp_lookahead_update_cadence)

    return optimizer


def get_sorted_params(model):
    param_count_and_name = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            param_count_and_name.append((p.numel(), n))

    pprint.pprint(sorted(param_count_and_name, reverse=True))


def count_parameters(model):
    r"""
    Due to Federico Baldassarre
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
