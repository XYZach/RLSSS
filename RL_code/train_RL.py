import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import copy

import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn

from networks.vnet import VNet
from utils import util
from dataloaders.dataloader import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/raid/zxy/datasets/2018LA_Seg_Training Set', help='Name of Experiment')
parser.add_argument('--list_folder', type=str, default='../data', help='list folder')
parser.add_argument('--exp', type=str,  default='LAHeart_RL_pretrain_batch4', help='model_name')  # 跑前修改
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=1e-2, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--num_classes', type=int,  default=2, help='number of class')
parser.add_argument('--data_num', type=int,  default=80, help='number of training data')
parser.add_argument('--labeled_num', type=int,  default=16, help='number of labeled data')
parser.add_argument('--patch_size', type=tuple,  default=(112, 112, 80), help='patch_size')
parser.add_argument('--normalization', type=str,  default='batchnorm', help='groupnorm or batchnorm')
parser.add_argument('--use_pretrain', type=bool,  default=True, help='pretrain')  # for teacher model
parser.add_argument('--optimizer', type=str,  default='sgd', help='optimizer')
parser.add_argument('--lr_change_per_iteration', type=int,  default=2500, help='lr_change_per_iteration')
parser.add_argument('--save_per_iteration', type=int,  default=1000, help='save per iteration')
args = parser.parse_args()

train_data_path = args.root_path  # the path of h5 file
snapshot_path = os.path.join('../model/', args.exp+'_labeled'+str(args.labeled_num))  # save path
pretrain_model_path = f'/home/zxy/codes/UA-MT/model/LAHeart_labeled{args.labeled_num}_vnet_supervisedonly_baseline_batchnorm/iter_6000.pth'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations  # 6000
base_lr = args.base_lr  # 1e-2
labeled_bs = args.labeled_bs
data_num = args.data_num
labeled_num = args.labeled_num
num_classes = args.num_classes
patch_size = args.patch_size

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def main():
    # make logger file
    util.makedir(snapshot_path)

    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # save this .py
    py_path_old = sys.argv[0]
    py_path_new = os.path.join(snapshot_path, os.path.basename(py_path_old))
    shutil.copy(py_path_old, py_path_new)

    def create_model(use_pretrain=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization=args.normalization, has_dropout=True)
        model = net.cuda()
        if use_pretrain:
            logging.info('pretrain_model_path:' + pretrain_model_path)
            model.load_state_dict(torch.load(pretrain_model_path))
        return model

    model_student = create_model()
    model_teacher = create_model(use_pretrain=args.use_pretrain)

    db_train = LAHeart(base_dir=train_data_path,
                       list_folder=args.list_folder,
                         split='train',
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(patch_size),
                             ToTensor()
                         ]))
    db_val = LAHeart(base_dir=train_data_path,
                     list_folder=args.list_folder,
                       split='test',
                       transform=transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    labeled_idxs = list(range(labeled_num))
    unlabeled_idxs = list(range(labeled_num, data_num))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    if args.optimizer == 'sgd':
        optimizer_student = optim.SGD(model_student.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        optimizer_teacher = optim.SGD(model_teacher.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif args.optimizer == 'adam':
        optimizer_student = optim.Adam(model_student.parameters(), lr=base_lr)
        optimizer_teacher = optim.Adam(model_student.parameters(), lr=base_lr)

    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr  # 0.01
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            model_student.train()
            model_teacher.train()
            time2 = time.time()
            print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # train student model on unlabeled data
            pred_unlabeled_student = model_student(volume_batch[labeled_bs:])
            pred_unlabeled_teacher = model_teacher(volume_batch[labeled_bs:])

            optimizer_student.zero_grad()
            pred_unlabeled_teacher_argmax = torch.argmax(pred_unlabeled_teacher, dim=1)

            loss_student_unlabeled = criterion(pred_unlabeled_student, pred_unlabeled_teacher_argmax)
            loss_student_unlabeled.backward()
            grad_student_unlabeled = copy.deepcopy([para.grad for para in model_student.parameters()])
            optimizer_student.step()

            optimizer_student.zero_grad()
            # cal student's performance(dice) on labeled data
            pred_labeled_student = model_student(volume_batch[:labeled_bs])

            loss_student_labeled = criterion(pred_labeled_student, label_batch[:labeled_bs])
            loss_student_labeled.backward()
            grad_student_labeled = copy.deepcopy([para.grad for para in model_student.parameters()])

            grad_t0mult1 = 0
            for i in range(len(grad_student_unlabeled)):
                grad_t0mult1 += torch.sum(torch.mul(grad_student_unlabeled[i], grad_student_labeled[i]))
            h = lr_ * grad_t0mult1

            # Compute the teacher’s gradient from the student’s feedback:
            loss_teacher_unlabeled = criterion(pred_unlabeled_teacher, pred_unlabeled_teacher_argmax)
            optimizer_teacher.zero_grad()
            loss_teacher_unlabeled.backward()
            grad_teacher_unlabeled = copy.deepcopy([h * para.grad for para in model_teacher.parameters()])

            optimizer_teacher.zero_grad()
            pred_labeled_teacher = model_teacher(volume_batch[:labeled_bs])  # labeled
            loss_teacher_labeled = criterion(pred_labeled_teacher, label_batch[:labeled_bs])
            loss_teacher_labeled.backward()
            grad_teacher_labeled = copy.deepcopy([para.grad for para in model_teacher.parameters()])

            # Update the teacher:
            for grad_index, para in enumerate(model_teacher.parameters()):
                para.grad = grad_teacher_unlabeled[grad_index] + grad_teacher_labeled[grad_index]
            optimizer_teacher.step()

            iter_num = iter_num + 1
            writer.add_scalar('loss/loss_student_unlabeled', loss_student_unlabeled.item(), iter_num)
            writer.add_scalar('loss/loss_student_labeled', loss_student_labeled.item(), iter_num)
            writer.add_scalar('loss/loss_teacher_labeled', loss_teacher_labeled.item(), iter_num)

            logging.info('iteration : %d loss_student_unlabeled : %f loss_student_labeled: %f, '
                         'loss_teacher_labeled: %f' %
                         (iter_num, loss_student_unlabeled.item(), loss_student_labeled.item(),
                          loss_teacher_labeled.item()))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 15:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(pred_labeled_student[0, :, :, :, 15:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = torch.from_numpy(image[:, np.newaxis, ...].repeat(3, axis=1))
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_student', grid_image, iter_num)

                image = torch.max(pred_labeled_teacher[0, :, :, :, 15:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = torch.from_numpy(image[:, np.newaxis, ...].repeat(3, axis=1))
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_teacher', grid_image, iter_num)

                image = label_batch[0, :, :, 15:61:10].permute(2, 0, 1).data.cpu().numpy()
                image = torch.from_numpy(image[:, np.newaxis, ...].repeat(3, axis=1))
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                #####
                image = volume_batch[-1, 0:1, :, :, 15:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(pred_unlabeled_student[-1, :, :, :, 15:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = torch.from_numpy(image[:, np.newaxis, ...].repeat(3, axis=1))
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label_student', grid_image, iter_num)

                image = torch.max(pred_unlabeled_teacher[-1, :, :, :, 15:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = torch.from_numpy(image[:, np.newaxis, ...].repeat(3, axis=1))
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label_teacher', grid_image, iter_num)

                image = label_batch[-1, :, :, 15:61:10].permute(2, 0, 1).data.cpu().numpy()
                image = torch.from_numpy(image[:, np.newaxis, ...].repeat(3, axis=1))
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            if iter_num % args.save_per_iteration == 0:
                save_model_student_path = os.path.join(snapshot_path, f'iter_{iter_num}_student.pth')
                save_model_teacher_path = os.path.join(snapshot_path, f'iter_{iter_num}_teacher.pth')
                torch.save(model_student.state_dict(), save_model_student_path)
                torch.save(model_teacher.state_dict(), save_model_teacher_path)

            ## change lr
            if iter_num % args.lr_change_per_iteration == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer_student.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_teacher.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    writer.close()


if __name__ == '__main__':
    main()



