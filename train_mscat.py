from __future__ import print_function
import os
import copy
import torch
import time
import argparse
import numpy as np
import logging
from torch import nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from resnet import ResNet18
from ema_model import EMAModel
from function import PurifyAdapter
from function import StochasticMLP
from function import AttnFusionUpDown
from function import SupConLoss
from mscat import mscat_loss
from dataset import CIFAR10
parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--train-batch-size', type=int, default=128, metavar='N', help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N', help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=130, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', default=8/255, help='radii of perturbation')
parser.add_argument('--step-size', default=2/255, help='perturb step size')
parser.add_argument('--train-num-steps', default=10, help='train perturb number of steps')
parser.add_argument('--test-num-steps', default=20, help='test perturb number of steps')
parser.add_argument('--theta1', type=int, default=5, metavar='S', help='logit align')
parser.add_argument('--theta2', type=int, default=1, metavar='S', help='group align')
parser.add_argument('--theta3', type=int, default=1, metavar='S', help='task align')
parser.add_argument('--theta4', type=int, default=0.1, metavar='S', help='task align')
parser.add_argument('--theta5', type=int, default=3, metavar='S', help='task align')
parser.add_argument('--k', type=int, default=7, metavar='S', help='task align')
parser.add_argument('--ema-epoch', default=76, type=int, help='Starting epoch of moving average')
parser.add_argument('--ema-decay', default=0.999, type=float, metavar='W')
parser.add_argument('--pac-epoch', default=75, type=int, help='Starting epoch of moving average')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./mscat', help='directory of model for saving checkpoint')
args = parser.parse_args()
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
makedir(args.model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
set_seed(args)
def sigmoid_rampup(alpha, current, start_es, end_es):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if current < start_es:
        return 0.0
    if current > end_es:
        return 1.0
    else:
        import math
        phase = 1.0 - (current - start_es) / (end_es - start_es)
        return math.exp(-alpha * phase * phase)
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 100:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
class Logger(object):
    def __init__(self, path):
        self.logger = logging.getLogger()
        self.path = path
        self.set_file_logger()
    def set_file_logger(self):
        handler = logging.FileHandler(self.path, 'w+')
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    def log(self, message):
        self.logger.info(message)
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])
train_dataset = CIFAR10(root='./cifar10_data', train=True, download=False, transform=transform_train)
test_dataset = CIFAR10(root='./cifar10_data', train=False, download=False, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
def train(args, model, move_average, purify_adapter, sample_adapter, fusion_adapter, train_loader, optimizer, contrastive,  epoch):
    model.train()
    start = time.time()
    robust_accuracy_total, natural_accuracy_total = 0, 0
    for batch_idx, (data, target, index) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss, robust_accuracy, natural_accuracy = mscat_loss(model=model,
                                                                               move_average=move_average,
                                                           purify_adapter=purify_adapter,
                                                           sample_adapter=sample_adapter, 
                                                           fusion_adapter=fusion_adapter, 
                                                           x_natural=data,
                                                           y=target,
                                                           optimizer=optimizer,
                                                           contrastive=contrastive,
                                                           step_size=args.step_size,
                                                           epsilon=args.epsilon,
                                                           perturb_steps=args.train_num_steps,
                                                           theta1=args.theta1,
                                                           theta2=args.theta2,
                                                           theta3=args.theta3,
                                                           theta4=args.theta4,
                                                           theta5=args.theta5,
                                                           k=args.k,
                                                           current_epoch=epoch,
                                                           pac_epoch=args.pac_epoch)

        loss.backward()
        optimizer.step()
        move_average.update(epoch, ema_epoch=args.ema_epoch, decay=args.ema_decay)
        robust_accuracy_total += robust_accuracy
        natural_accuracy_total += natural_accuracy
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    train_robust_accuracy = robust_accuracy_total / len(train_loader.dataset)
    train_natural_accuracy = natural_accuracy_total / len(train_loader.dataset)
    print('Training：train_natural_accuracy: {:.4f}, train_robust_accuracy: {:.4f}'.format(train_natural_accuracy, train_robust_accuracy))
    return train_natural_accuracy, train_robust_accuracy
def pgd_whitebox(model,
                  X,
                  y,
                  num_steps,
                  step_size=args.step_size,
                  epsilon=args.epsilon):
    out = model(X)[0]
    natural_accuracy = (out.data.max(1)[1] == y.data).float().sum()
    x_adv = X.detach() + torch.FloatTensor(*X.shape).uniform_(-epsilon, epsilon).cuda()
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv)[0], y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    robust_accuracy = (model(x_adv)[0].data.max(1)[1] == y.data).float().sum()
    return natural_accuracy, robust_accuracy
def eval_pgd_whitebox(model, test_loader, num_steps):
    model.eval()
    natural_accuracy_total, robust_accuracy_total = 0, 0
    for data, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)
        natural_accuracy, robust_accuracy = pgd_whitebox(model, X, y, num_steps)
        natural_accuracy_total += natural_accuracy
        robust_accuracy_total += robust_accuracy
    test_natural_accuracy = natural_accuracy_total / len(test_loader.dataset)
    test_robust_accuracy = robust_accuracy_total / len(test_loader.dataset)
    print('Testing Teacher： natural accuracy:{:.4f},  pgd accuracy:{:.4f}'.format(test_natural_accuracy, test_robust_accuracy))
    return test_natural_accuracy, test_robust_accuracy
def main(args):
    model = ResNet18(num_classes=10).to(device)
    purify_adapter = PurifyAdapter(dim=512).to(device)
    fusion_adapter = AttnFusionUpDown(in_dim=10, up_dim=512, head=8, hidden_ratio=0.25).to(device)
    sample_adapter = StochasticMLP(in_dim=512, hidden_dim=128, out_dim=10).to(device)
    optimizer = optim.SGD(list(model.parameters()) + list(purify_adapter.parameters()) + list(sample_adapter.parameters()) + list(fusion_adapter.parameters()),  
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    ema_model = copy.deepcopy(model)
    move_average = EMAModel(model=model, ema_model=ema_model, update_bn=True)
    contrastive = SupConLoss()
    start = time.time()
    logger = Logger(os.path.join(args.model_dir, 'mscat.log'))
    logger.log('Using device: {}'.format(device))
    logger.log('\n')
    logger.log('Training for {} epochs'.format(args.epochs))
    logger.log('\n')
    for epoch in range(1, args.epochs + 1):
        logger.log('=============Epoch {}=============='.format(epoch))
        train_test_start = time.time()
        adjust_learning_rate(args, optimizer, epoch)
        print('=============================Training Epoch {}================================='.format(epoch))
        train_natural_accuracy, train_robust_accuracy = train(args, model, move_average, purify_adapter, sample_adapter, fusion_adapter, train_loader, optimizer, contrastive, ema_logit, epoch)
        train_end = time.time()
        logger.log(
            'Training: Robust Accuracy: {:.4f}.\tNatural Accuracy: {:.4f}.\tLR: {:.4f}.\tTime taken: {:.4f}'
            .format(train_robust_accuracy, train_natural_accuracy,
                    optimizer.param_groups[0]['lr'], (train_end - train_test_start) / 60))
        print('==============================Testing Epoch {}================================'.format(epoch))
        test_natural_accuracy, test_robust_accuracy = eval_pgd_whitebox(model, test_loader, num_steps=args.test_num_steps)
        test_end = time.time()
        logger.log('Testing Teacher: Natural Accuracy: {:.4f}.\tRobust Accuracy: {:.4f}\tTime taken:{:.4f}'
                   .format(test_natural_accuracy, test_robust_accuracy, (test_end - train_test_start) / 60))
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'mscat_{}.py'.format(epoch)))
    end = time.time()
    print('Final time: %.3f' % ((end - start) / 60) + ' min')
    logger.log('\n')
    logger.log('Total time: {}'.format((end - start) / 60))
    logger.log('Script Completed')

    return

if __name__ == '__main__':
    main(args)

