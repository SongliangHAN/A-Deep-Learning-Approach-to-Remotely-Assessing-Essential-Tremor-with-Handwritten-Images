import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
from PIL import Image
from resnet50 import Bottleneck
from resnet50 import ResNet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import namedtuple
import random


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_dir='ET分级大圆new/train'
test_dir='ET分级大圆new/test'
train_data = datasets.ImageFolder(root = train_dir,
                                  transform = transforms.ToTensor())

pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                        #    transforms.RandomRotation(5),
                        #    transforms.RandomHorizontalFlip(0.5),
                        #    transforms.RandomCrop(pretrained_size, padding = 10),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means,
                                                std = pretrained_stds)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.CenterCrop(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means,
                                                std = pretrained_stds)
                       ])


train_data = datasets.ImageFolder(root = train_dir,
                                  transform = train_transforms)

test_data = datasets.ImageFolder(root = test_dir,
                                 transform = test_transforms)

VALID_RATIO = 0.5

n_test_examples = int(len(test_data) * VALID_RATIO)
n_valid_examples = len(test_data) - n_test_examples

test_data, valid_data = data.random_split(test_data,
                                           [n_test_examples, n_valid_examples])

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

BATCH_SIZE = 64

train_iterator = data.DataLoader(train_data,
                                 shuffle = True,
                                 batch_size = BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data,
                                 batch_size = BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size = BATCH_SIZE)
#################################################################Densenet#############################################
model = models.densenet121(pretrained=True)
num_classes = 5
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, num_classes)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 解冻所有层以应用分层学习率
for param in model.parameters():
    param.requires_grad = True

print(f"Total trainable densenet parameters: {count_trainable_parameters(model):,}")

base_lr = 0.001
step_size = 3
decay_factor = 0.5
param_groups = get_param_groups_step_decay(model, base_lr=base_lr, step_size=step_size, decay_factor=decay_factor)
optimizer = torch.optim.Adam(param_groups)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print('DenseNet')
train_model(model, train_iterator, valid_iterator, criterion, optimizer, num_epochs=10)
test_loss, test_acc_1, test_acc_5 = evaluate_baseline(model, test_iterator, criterion, device)

print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ' \
      f'Test Acc @5: {test_acc_5*100:6.2f}%')


#########################################################MobileV2############################################################
model = models.mobilenet_v2(pretrained=True)
num_classes=5
num_features = model.last_channel  # MobileNetV2 最后一层的输入特征数量
model.classifier[1] = nn.Linear(num_features, num_classes)

# 基础学习率、步长、衰减因子
base_lr = 0.001
step_size = 3
decay_factor = 0.5

# 获取分组学习率参数
param_groups = get_param_groups_step_decay(model, base_lr=base_lr, step_size=step_size, decay_factor=decay_factor)

# 定义优化器
optimizer = torch.optim.Adam(param_groups)

# 验证学习率分布
for i, param_group in enumerate(optimizer.param_groups):
    print(f"Layer Group {i+1}: Learning Rate = {param_group['lr']}")
print('MobileNet')
train_model(model, train_iterator, valid_iterator, criterion, optimizer, num_epochs=10)
test_loss, test_acc_1, test_acc_5 = evaluate_baseline(model, test_iterator, criterion, device)
print(f"Total trainable mobilenet parameters: {count_trainable_parameters(model):,}")
print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ' \
      f'Test Acc @5: {test_acc_5*100:6.2f}%')

########################################################ConvNextTiny########################################

model = models.convnext_tiny(pretrained=True)
num_classes = 5
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, num_classes)

# 定义分层学习率优化器
param_groups = get_param_groups_step_decay(model, base_lr=0.001, step_size=3, decay_factor=0.5)
optimizer = torch.optim.Adam(param_groups)
model = model.to(device)

print('ConvNextTiny')
train_model(model, train_iterator, valid_iterator, criterion, optimizer, num_epochs=10)
test_loss, test_acc_1, test_acc_5 = evaluate_baseline(model, test_iterator, criterion, device)
print(f"Total trainable convnext parameters: {count_trainable_parameters(model):,}")
print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ' \
      f'Test Acc @5: {test_acc_5*100:6.2f}%')