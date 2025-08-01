import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets, hopenet, hopelessnet
import torch.utils.model_zoo as model_zoo

import os

# 设置modelzoo下载路径为checkpoints目录
def setup_modelzoo_download_path():
    """设置modelzoo下载路径为checkpoints目录"""
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 设置torch hub的缓存目录
    os.environ['TORCH_HOME'] = checkpoint_dir
    # 设置modelzoo的缓存目录
    torch.hub.set_dir(checkpoint_dir)
    
    return checkpoint_dir

def setup_device(gpu_id):
    """设置GPU设备，如果gpu_id为-1则使用所有GPU"""
    if gpu_id == -1:
        # 使用所有可用的GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            num_gpus = torch.cuda.device_count()
            print(f'使用所有GPU: {num_gpus}个GPU可用')
            return device, num_gpus
        else:
            print('警告: 没有可用的GPU，将使用CPU')
            return torch.device('cpu'), 0
    else:
        # 使用指定的GPU
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{gpu_id}')
            print(f'使用GPU: {gpu_id}')
            return device, 1
        else:
            print(f'警告: GPU {gpu_id}不可用，将使用CPU')
            return torch.device('cpu'), 0

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument(
        '--gpu_id', dest='gpu_id', help='GPU device id to use [0], use -1 for all GPUs',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', 
        help='Maximum number of training epochs.',
        default=50, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=64, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.000001, type=float)
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.', 
        default='Pose_300W_LP', type=str)
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)
    parser.add_argument(
        '--filename_list', dest='filename_list', 
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str)
    parser.add_argument(
        '--output_string', dest='output_string', 
        help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--arch', dest='arch', 
        help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], '
            'ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def get_ignored_params(model, arch):
    # Generator function that yields ignored params.
    # 处理DataParallel的情况
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
        
    if arch.find('ResNet') >= 0:
        b = [actual_model.conv1, actual_model.bn1, actual_model.fc_finetune]
    elif arch.find('Squeezenet') >= 0 or arch.find('MobileNetV2') >= 0:
        b = [actual_model.features[0]]
    else:
        raise('Invalid architecture is passed!')

    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model, arch):
    # Generator function that yields params that will be optimized.
    # 处理DataParallel的情况
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
        
    if arch.find('ResNet') >= 0:
        b = [actual_model.layer1, actual_model.layer2, actual_model.layer3, actual_model.layer4]
    elif arch.find('Squeezenet') >= 0 or arch.find('MobileNetV2') >= 0:
        b = [actual_model.features[1:]]
    else:
        raise('Invalid architecture is passed!')

    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model, arch):
    # Generator function that yields fc layer params.
    # 处理DataParallel的情况
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
        
    if arch.find('ResNet') >= 0:
        b = [actual_model.fc_yaw, actual_model.fc_pitch, actual_model.fc_roll]
    elif arch.find('Squeezenet') >= 0 or arch.find('MobileNetV2') >= 0:
        b = [
            actual_model.classifier_yaw, 
            actual_model.classifier_pitch, 
            actual_model.classifier_roll
        ]
    else:
        raise('Invalid architecture is passed!')

    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()

    # 设置modelzoo下载路径
    checkpoint_dir = setup_modelzoo_download_path()
    print(f'Modelzoo下载路径设置为: {checkpoint_dir}')

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    
    # 设置设备
    device, num_gpus = setup_device(args.gpu_id)

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # Network architecture
    if args.arch == 'ResNet18':
        model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif args.arch == 'ResNet34':
        model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [3,4,6,3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif args.arch == 'ResNet101':
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif args.arch == 'ResNet152':
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    elif args.arch == 'Squeezenet_1_0':
        model = hopelessnet.Hopeless_Squeezenet(args.arch, 66)
        pre_url = \
            'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth'
    elif args.arch == 'Squeezenet_1_1':
        model = hopelessnet.Hopeless_Squeezenet(args.arch, 66)
        pre_url = \
            'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth'
    elif args.arch == 'MobileNetV2':
        model = hopelessnet.Hopeless_MobileNetV2(66, 1.0)
        pre_url = \
            'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
    else:
        if args.arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    if args.snapshot == '':
        print(f'从modelzoo下载预训练模型到: {checkpoint_dir}')
        load_filtered_state_dict(model, model_zoo.load_url(pre_url))
    else:
        print(f'加载本地模型: {args.snapshot}')
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)

    print('Loading data...')

    transformations = transforms.Compose([transforms.Resize(240),
            transforms.RandomCrop(224), transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
            )
        ])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Synhead':
        pose_dataset = datasets.Synhead(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset   == 'AFW':
        pose_dataset = datasets.AFW(
            args.data_dir, args.filename_list, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()

    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

    # 将模型移动到设备并设置数据并行
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    reg_criterion = nn.MSELoss().to(device)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax().to(device)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).to(device)

    optimizer = torch.optim.Adam([
        {'params': get_ignored_params(model, args.arch), 'lr': 0},
        {'params': get_non_ignored_params(model, args.arch), 'lr': args.lr},
        {'params': get_fc_params(model, args.arch), 'lr': args.lr * 5}
        ], lr = args.lr)

    print('Ready to train network.')
    for epoch in range(num_epochs):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).to(device)

            # Binned labels
            label_yaw = Variable(labels[:,0]).to(device)
            label_pitch = Variable(labels[:,1]).to(device)
            label_roll = Variable(labels[:,2]).to(device)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).to(device)
            label_pitch_cont = Variable(cont_labels[:,1]).to(device)
            label_roll_cont = Variable(cont_labels[:,2]).to(device)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = \
                torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = \
                torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = \
                torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = \
                [torch.tensor(1.0).to(device) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: '
                    'Yaw %.4f, Pitch %.4f, Roll %.4f'%(
                        epoch+1, 
                        num_epochs, 
                        i+1, 
                        len(pose_dataset)//batch_size, 
                        loss_yaw.item(), 
                        loss_pitch.item(), 
                        loss_roll.item()
                    )
                )

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...')
            # 如果使用DataParallel，需要保存model.module的状态
            if num_gpus > 1:
                torch.save(model.module.state_dict(),
                'output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')
            else:
                torch.save(model.state_dict(),
                'output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')
