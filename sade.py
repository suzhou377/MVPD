import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.function import mixup
from torch.autograd import Variable
def mvpd_loss(model, 
             semantic_explorer, 
              logit_explorer,
                 fusion_explorer,
                 x_natural,
                y,
                optimizer,
                contrastive,
                ema_logit,
                step_size=2/255,
                epsilon=8/255,
                perturb_steps=10,
                theta1=5,
                theta2=1,
                theta3=1,
                theta4=0.1,
                k=7,
                epoch=75,
                mvpd_epoch=75):
    model.eval()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    # generate adversarial example
    x_adv = x_natural.detach() + torch.FloatTensor(*x_natural.shape).uniform_(-epsilon, epsilon).cuda()
    for index in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_pgd = F.cross_entropy( model(x_adv)[0], y)
        grad = torch.autograd.grad(loss_pgd, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    '''student'''
    nat_logit, nat_emb = model(x_natural)
    adv_logit, adv_emb = model(x_adv)
    '''accuracy'''
    nat_accuracy = (torch.argmax(nat_logit, dim=1) == y).sum().item()
    adv_accuracy = (torch.argmax(adv_logit, dim=1) == y).sum().item()
    if epoch >= mvpd_epoch:
        explorer_up_nat, explorer_down_nat = semantic_explorer(nat_emb)
        explorer_up_adv, explorer_down_adv = semantic_explorer(adv_emb)
        semantic_loss = (contrastive(explorer_up_nat, explorer_up_adv, labels=y) + contrastive(explorer_down_nat, explorer_down_adv, labels=y) + contrastive(nat_emb, adv_emb, labels=y)) / 3
        explorer_fuse_nat, explorer_fuse_adv = (explorer_up_nat + explorer_down_nat) / 2, (explorer_up_adv + explorer_down_adv) / 2
        explorer_nat_logit, mu_nat, logvar_nat = logit_explorer(explorer_fuse_nat, n_samples=k)
        explorer_adv_logit, mu_adv, logvar_adv = logit_explorer(explorer_fuse_adv, n_samples=k)             
        dis_loss = (logit_explorer.kl_loss(mu_nat, logvar_nat) + logit_explorer.kl_loss(mu_adv, logvar_adv)) / 2
        nat_fusion, adv_fusion = fusion_explorer(nat_logit, explorer_nat_logit), fusion_explorer(adv_logit, explorer_adv_logit)
        if k > 1:
            explorer_nat_logit, explorer_adv_logit = explorer_nat_logit.mean(dim=1), explorer_adv_logit.mean(dim=1)
        else:
            pass
        logit_loss = (F.mse_loss(nat_logit, nat_fusion) + F.mse_loss(explorer_nat_logit, nat_fusion) + 
                     F.mse_loss(adv_logit, adv_fusion) + F.mse_loss(explorer_adv_logit, adv_fusion)) / 4
        '''cross entropy'''
        ce_loss = F.cross_entropy(adv_logit, y)
        '''align loss'''
        align_loss = criterion_kl(F.log_softmax(adv_logit, dim=1), F.softmax(nat_logit, dim=1))
        '''total loss'''
        loss = ce_loss + theta1 * align_loss + theta2 * semantic_loss + theta3 * logit_loss + theta4 * dis_loss
        return loss, adv_accuracy, nat_accuracy
    else:
        loss = F.cross_entropy(adv_logit, y)
    return loss, adv_accuracy, nat_accuracy
