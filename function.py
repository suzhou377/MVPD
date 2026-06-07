import torch
import math
import random
import torchvision
import torch.nn as nn
import numpy as np
import torch.fft as fft
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from torch.distributions.bernoulli import Bernoulli

class FusionMLP(nn.Module):
    def __init__(self, logit_dim, hidden_dim=128, act=nn.GELU):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(logit_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act(),
            nn.Linear(hidden_dim, logit_dim)
        )
    def forward(self, logit1, logit2, logit3):
        concat_logits = torch.cat([logit1, logit2, logit3], dim=1)
        fused_logit = self.mlp(concat_logits)
        return fused_logit
def BSL(labels, logits, sample_per_class):
    spc = torch.tensor(sample_per_class).type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + (spc + 1).log()
    loss = F.cross_entropy(input=logits, target=labels)
    return loss


class PurifyAdapter(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.up_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x)[0]
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.up_ffn(x)
        x = x + residual
        return x



    
class UpAdapter(nn.Module):
    def __init__(self, in_dim=10, up_dim=512, hidden_ratio=0.25, act=nn.GELU): 
        super().__init__()
        hidden = int(up_dim * hidden_ratio)     
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), act(), nn.Linear(hidden, up_dim))
    def forward(self, x):
        return self.net(x)

class DownAdapter(nn.Module):
    def __init__(self, up_dim=512, out_dim=10, hidden_ratio=0.25, act=nn.GELU):  
        super().__init__()
        hidden = int(up_dim * hidden_ratio)      
        self.net = nn.Sequential(nn.Linear(up_dim, hidden), act(), nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.net(x) 
class AttnFusionUpDown(nn.Module):
    def __init__(self, in_dim=10, up_dim=512, head=8, hidden_ratio=0.25): 
        super().__init__()
        self.up_logit = UpAdapter(in_dim, up_dim, hidden_ratio, nn.GELU)
        self.up_group = UpAdapter(in_dim, up_dim, hidden_ratio, nn.GELU)
        self.cross = nn.MultiheadAttention(up_dim, head, batch_first=True)
        self.down = DownAdapter(up_dim, in_dim, hidden_ratio, nn.GELU)
    def forward(self, logit, group):
        q = self.up_logit(logit).unsqueeze(1)       
        if group.dim() == 2:                           
            group = group.unsqueeze(1)
        kv = self.up_group(group)                    
        out, _ = self.cross(q, kv, kv)              
        out = self.down(out).squeeze(1)              
        return out

class StochasticMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * out_dim))
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std         
    def forward(self, x, n_samples=1):
        out = self.net(x)                     
        mu, logvar = out.chunk(2, dim=1)      
        logvar = torch.clamp(logvar, min=-10, max=10)
        if n_samples > 1:
            mu  = mu.unsqueeze(1).expand(-1, n_samples, -1)   
            logvar = logvar.unsqueeze(1).expand(-1, n_samples, -1)
            sample = self.reparameterize(mu, logvar)       
            return sample, mu.mean(1), logvar.mean(1)     
        else:
            sample = self.reparameterize(mu, logvar)        
            return sample, mu, logvar
    @staticmethod
    def kl_loss(mu, logvar):
        return -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.2, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
    def forward(self, view_one, view_two, labels=None, mask=None):
        features = torch.cat([view_one.unsqueeze(1), view_two.unsqueeze(1)], dim=1)
        features = F.normalize(features, p=2, dim=2) 
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],' 'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:  
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None: 
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: 
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(input=torch.ones_like(mask), dim=1, index=torch.arange(batch_size * anchor_count).view(-1, 1).to(device), value=0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=mask_pos_pairs.dtype, device=mask_pos_pairs.device), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        loss = - mean_log_prob_pos.view(anchor_count, batch_size).mean()
        return loss


