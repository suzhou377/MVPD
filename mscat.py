import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def mscat_loss(model,
                 move_average,
                 purify_adapter,
                 sample_adapter,
                 fusion_adapter,
                 x_natural,
                y,
                contrastive,
                step_size=2/255,
                epsilon=8/255,
                perturb_steps=10,
                theta1=5,
                theta2=1,
                theta3=1,
                theta4=0.1,
                theta5=3,
                k=7,
                current_epoch=75,
                pac_epoch=75):
    model.eval()
    x_adv = x_natural.detach() + torch.FloatTensor(*x_natural.shape).uniform_(-epsilon, epsilon).cuda()
    for index in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
                nat_logit, adv_logit = model(x_natural)[0], model(x_adv)[0]
                loss_pgd = F.cross_entropy(adv_logit, y) + theta5 * F.mse_loss(adv_logit, nat_logit)
        grad = torch.autograd.grad(loss_pgd, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    natural_logit, natural_emb = model(x_natural)
    robust_logit, robust_emb = model(x_adv)
    with torch.no_grad():
        move_natural_emb, move_robust_emb = move_average(x_natural)[1], move_average(x_adv)[1]
    natural_accuracy = (torch.argmax(natural_logit, dim=1) == y).sum().item()
    robust_accuracy = (torch.argmax(robust_logit, dim=1) == y).sum().item()
    if current_epoch >= pac_epoch:
        purify_natural_emb, purify_robust_emb = purify_adapter(natural_emb), purify_adapter(robust_emb)
        purify_move_natural_emb, purify_move_robust_emb = purify_adapter(move_natural_emb), purify_adapter(move_robust_emb)
        purify_loss = (contrastive(purify_natural_emb, purify_robust_emb, labels=y) + contrastive(move_natural_emb, move_robust_emb, labels=y) +
                        contrastive(purify_move_natural_emb, purify_move_robust_emb, labels=y) + contrastive(natural_emb, robust_emb, labels=y)) / 4
        mean_natural_emb = (purify_natural_emb + move_natural_emb + purify_move_natural_emb + natural_emb) / 4
        mean_robust_emb = (purify_robust_emb + move_robust_emb + purify_move_robust_emb + robust_emb) / 4
        with torch.no_grad():
            sample_natural_logit, mu_nat, logvar_nat = sample_adapter(mean_natural_emb, n_samples=k)
            sample_robust_logit, mu_rob, logvar_rob = sample_adapter(mean_robust_emb, n_samples=k)
        dis_loss = (sample_adapter.kl_loss(mu_nat, logvar_nat) + sample_adapter.kl_loss(mu_rob, logvar_rob)) / 2
        natural_fusion, robust_fusion = fusion_adapter(natural_logit, sample_natural_logit), fusion_adapter(robust_logit, sample_robust_logit)
        sample_natural_logit, sample_robust_logit = sample_natural_logit.mean(dim=1), sample_robust_logit.mean(dim=1)
        coll_loss = (F.mse_loss(natural_logit, natural_fusion) + F.mse_loss(sample_natural_logit, natural_fusion) +
                     F.mse_loss(robust_logit, robust_fusion) + F.mse_loss(sample_robust_logit, robust_fusion)) / 4
        ce_loss = F.cross_entropy(robust_logit, y)
        align_loss = F.kl_div(F.log_softmax(robust_logit, dim=1), F.softmax(natural_logit, dim=1), reduction='batchmean')
        loss = ce_loss + theta1 * align_loss + theta2 * purify_loss + theta3 * coll_loss + theta4 * dis_loss
        return loss, robust_accuracy, natural_accuracy