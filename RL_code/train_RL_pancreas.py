"""
Training rl
"""
# external imports
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import sys
import time
import datetime
import codecs
import logging
import random
import shutil
from tqdm import tqdm
import numpy as np
from medpy import metric
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.nn import BCEWithLogitsLoss, MSELoss

# internal imports
from utils import losses
from utils import util
from utils.losses import dice_loss
from dataloaders.Pancreas import Pancreas
from dataloaders.Pancreas import RandomNoise, ToTensor, RandomCrop, TwoStreamBatchSampler, CenterCrop
from networks.UNet3D import UNet3D
from networks.vnet import VNet

# parasr
parser = argparse.ArgumentParser()
parser.add_argument("--backbone",
                    type=str,
                    default='VNet',
                    choices=['UNet', 'VNet'],
                    help='backbone')

parser.add_argument("--gpu",
                    type=str,
                    default='0',
                    help="gpu id")

parser.add_argument("--batch_size",
                    type=int,
                    default='9',
                    help="batch size")

parser.add_argument("--labeled_bs",
                    type=int,
                    default=4,
                    help="labeled batch size per gpu")

parser.add_argument("--seed1",
                    type=int,
                    default='1124', 
                    help="seed 1")

parser.add_argument("--iter_from",
                    type=int,
                    dest="iter_from",
                    default=0,
                    help="iteration number to start training from")

parser.add_argument("--n_total_iter_from",
                    type=int,
                    dest="n_total_iter_from",
                    default=0,
                    help="iteration number(used for continuing training)")

parser.add_argument("--max_iterations",
                    type=int,
                    dest="max_iterations",
                    default=20000,
                    help="number of iterations of training")

parser.add_argument("--lr",
                    type=float,
                    dest="lr",
                    default=0.001,
                    help="adam: learning rate")

parser.add_argument("--n_save_iter",
                    type=int,
                    dest="n_save_iter",
                    default=500,
                    help="Save the model every time")

parser.add_argument("--data_path",
                    type=str,
                    dest="data_path",
                    default='/home/hra/dataset/Pancreas/Pancreas_region',
                    help="data path")

parser.add_argument("--fold_root_path",
                    type=str,
                    dest="fold_root_path",
                    default='../data/',
                    help="data path")

parser.add_argument("--model_dir_root_path",
                    type=str,
                    dest="model_dir_root_path",
                    default='../model/Pancreas/RL/',
                    help="root path to save the RL model")

parser.add_argument("--teacher_model_path",
                    type=str,
                    dest="teacher_model_path",
                    default='../model/Pancreas/VNet/2021-01-26/14:56:43/fold-6/model_4000.pth',
                    # default=None,
                    help="Path to pre-trained teacher model")

parser.add_argument("--note",
                    type=str,
                    dest="note",
                    default="RL",
                    # default=None,
                    help="note")

arg = parser.parse_args()


def train(backbone,
          gpu,
          batch_size,
          labeled_bs,
          seed1,
          iter_from,
          n_total_iter_from,
          max_iterations,
          lr,
          n_save_iter,
          data_path,
          fold_root_path,
          model_dir_root_path,
          teacher_model_path,
          note):
    """
    :param backbone
    :param gpu: gpu id
    :param batch_size: batch size
    :param labeled_bs: batch size for labeled data
    :param seed1: seed 1
    :param iter_from: iter_from to start training from
    :param n_total_iter_from: used for continuing training
    :param max_iterations: number of training max_iterations
    :param lr: learning rate
    :param n_save_iter: Determines how many epochs before saving model version
    :param data_path: data path
    :param fold_root_path: fold root path
    :param model_dir_root_path: the model directory root path to save to
    :param teacher_model_path: Path to pre-trained model
    :param note:
    :return:
    """

    """ setting """
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(seed1)
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)

    # time
    now = time.localtime()
    now_format = time.strftime("%Y-%m-%d %H:%M:%S", now)  # time format
    date_now = now_format.split(' ')[0]
    time_now = now_format.split(' ')[1]

    # save model path
    save_path = os.path.join(model_dir_root_path, date_now, time_now)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # print setting
    print("----------------------------------setting-------------------------------------")
    print("lr:%f" % lr)
    if teacher_model_path is None:
        print("pre-trained dir is None")
    else:
        print("pre-trained model path:%s" % teacher_model_path)
    print("path of saving model:%s" % save_path)
    print("----------------------------------setting-------------------------------------")

    # save parameters to TXT.
    parameter_dict = {"fold": fold_index,
                      "gpu": gpu,
                      "batch size": batch_size,
                      "labeled batch size": labeled_bs,
                      "model_pre_trained_dir": teacher_model_path,
                      "data path": data_path,
                      "iter_from": iter_from,
                      "lr": lr,
                      "save_path": save_path,
                      'note': note}
    txt_name = 'parameter_log.txt'
    path = os.path.join(save_path, txt_name)
    with codecs.open(path, mode='a', encoding='utf-8') as file_txt:
        for key, value in parameter_dict.items():
            file_txt.write(str(key) + ':' + str(value) + '\n')

    # save this .py
    py_path_old = sys.argv[0]
    py_path_new = os.path.join(save_path, os.path.basename(py_path_old))
    shutil.copy(py_path_old, py_path_new)

    # logging
    logging.basicConfig(filename=save_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(parameter_dict)

    # tensorboardX
    writer = SummaryWriter(log_dir=save_path)

    # label_dict
    label_list = [0, 1]

    # patch size
    patch_size = (96, 96, 96)

    """ data generator """
    # load all data path
    # todo: change your path
    train_volume_path, test_volume_path = .......
    train_label_path = [i.replace('img', 'label') for i in train_volume_path]
    test_label_path = [i.replace('img', 'label') for i in test_volume_path]

    # 62 data -> 12 data for labeled
    # 62 data -> 50 data for unlabeled
    labeled_idxs = list(range(12))
    unlabeled_idxs = list(range(12, 62))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    # dataset
    # training
    train_dataset = Pancreas(train_volume_path, train_label_path,
                             transform=transforms.Compose([RandomCrop(patch_size), RandomNoise(), ToTensor()]))
    eval_dataset = Pancreas(test_volume_path, test_label_path,
                            transform=transforms.Compose([CenterCrop(patch_size), ToTensor()]))

    # dataloader
    def worker_init_fn(worker_id):
        random.seed(seed1 + worker_id)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    """ model, optimizer, loss """
    if backbone == 'UNet':
        model_student = UNet3D(1, 2, has_dropout=False).cuda()
        model_teacher = UNet3D(1, 2, has_dropout=False).cuda()
    elif backbone == 'VNet':
        model_student = VNet(n_channels=1, n_classes=2, has_dropout=False, normalization='batchnorm').cuda()
        model_teacher = VNet(n_channels=1, n_classes=2, has_dropout=False, normalization='batchnorm').cuda()
    model_teacher.load_state_dict(torch.load(teacher_model_path))

    optimizer_student = torch.optim.Adam(model_student.parameters(), lr=lr)
    optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr=lr)
    # optimizer_student = torch.optim.SGD(model_student.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    # optimizer_teacher = torch.optim.SGD(model_teacher.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    criterion1 = nn.CrossEntropyLoss()

    """ training loop """
    n_total_iter = 0
    max_epoch = max_iterations // len(train_dataloader) + 1
    if n_total_iter_from != 0:
        n_total_iter = n_total_iter_from
    lr_ = lr

    best_eval_dice = 0
    for epoch in range(max_epoch):
        pseudo_label_avg_dice = 0
        student_pred_avg_dice = 0

        for i_batch, sampled_batch in enumerate(train_dataloader):
            # start_time
            start = time.time()

            # zero grad
            optimizer_student.zero_grad()
            optimizer_teacher.zero_grad()

            # generate moving data
            volume_batch = sampled_batch['volume'].to('cuda').float()
            seg_batch = sampled_batch['label'].to('cuda').long()

            # fetch data
            # student model run unlabeled data
            pred_unlabeled_student = model_student(volume_batch[labeled_bs:])
            # teacher model run unlabeled data
            pred_unlabeled_teacher = model_teacher(volume_batch[labeled_bs:])

            # zeros grad, compute loss, compute grad
            # Update the student using the pseudo label
            pred_unlabeled_teacher_argmax = torch.argmax(pred_unlabeled_teacher, dim=1)
            loss_student_unlabeled = criterion1(pred_unlabeled_student, pred_unlabeled_teacher_argmax)
            loss_student_unlabeled.backward()
            grad_student_t0 = copy.deepcopy([para.grad for para in model_student.parameters()])
            optimizer_student.step()

            optimizer_student.zero_grad()
            pred_labeled_student = model_student(volume_batch[:labeled_bs])
            loss_student_labeled = criterion1(pred_labeled_student, seg_batch[:labeled_bs, 0])
            loss_student_labeled.backward()
            grad_student_t1 = [para.grad for para in model_student.parameters()]

            # compute h
            # Compute the teacher’s feedback coefficient as in Equation 12
            grad_t0mult1 = 0
            for i in range(len(grad_student_t0)):
                grad_t0mult1 += torch.sum(torch.mul(grad_student_t0[i], grad_student_t1[i]))
            h = lr_ * grad_t0mult1

            # Compute the teacher’s gradient from the student’s feedback:
            loss_teacher_unlabeled = criterion1(pred_unlabeled_teacher, pred_unlabeled_teacher_argmax)
            optimizer_teacher.zero_grad()
            loss_teacher_unlabeled.backward()
            grad_teacher_unlabeled = [h * para.grad for para in model_teacher.parameters()]
            optimizer_teacher.zero_grad()
            pred_labeled_teacher = model_teacher(volume_batch[:labeled_bs])
            loss_teacher_labeled = criterion1(pred_labeled_teacher, seg_batch[:labeled_bs, 0])
            loss_teacher_labeled.backward()
            grad_teacher_labeled = [para.grad for para in model_teacher.parameters()]

            # Update the teacher:
            for para_index, para in enumerate(model_teacher.parameters()):
                para.grad = grad_teacher_unlabeled[para_index] + grad_teacher_labeled[para_index]
            optimizer_teacher.step()

            # dice of labeled data on student model
            seg_output_softmax_labeled = F.softmax(pred_labeled_student, dim=1)
            seg_output_softmax_labeled_argmax = torch.argmax(seg_output_softmax_labeled, dim=1)
            dice_value_labeled = metric.binary.dc(seg_output_softmax_labeled_argmax.cpu().detach().numpy(),
                                                  seg_batch[:labeled_bs, 0].cpu().detach().numpy())
            student_pred_avg_dice += dice_value_labeled
            seg_output_softmax_unlabeled = F.softmax(pred_unlabeled_teacher, dim=1)
            seg_output_softmax_unlabeled_argmax = torch.argmax(seg_output_softmax_unlabeled, dim=1)
            dice_value_unlabeled = metric.binary.dc(seg_output_softmax_unlabeled_argmax.cpu().detach().numpy(),
                                                    seg_batch[labeled_bs:, 0].cpu().detach().numpy())
            pseudo_label_avg_dice += dice_value_unlabeled

            # ---------------------
            #     Print log
            # ---------------------
            n_total_iter += 1
            # Determine approximate time left
            end = time.time()
            iter_left = (max_epoch - epoch) * (len(train_dataloader) - i_batch)
            time_left = datetime.timedelta(seconds=iter_left * (end - start))

            # print log
            logging.info("[Epoch: %4d/%d] [n_total_iter: %5d] [Total index: %2d/%d] "
                         "[Dice-labeled: %f] [Dice-unlabeled: %f] [ETA: %s]"
                         % (epoch, max_epoch, n_total_iter, i_batch+1, len(train_dataloader),
                            dice_value_labeled, dice_value_unlabeled, time_left))

            # tensorboardX log writer
            writer.add_scalar('lr', lr_, n_total_iter)
            # writer.add_scalar('Dice/labeled', dice_value_labeled, n_total_iter)
            # writer.add_scalar('Dice/unlabeled', dice_value_unlabeled, n_total_iter)

            if n_total_iter % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 30:71:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('Image', grid_image, n_total_iter)

                image = seg_batch[0, 0:1, :, :, 30:71:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('Groundtruth/segmentation_label', grid_image, n_total_iter)

                image = torch.argmax(pred_labeled_student, dim=1)[0:1, :, :, 30:71:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('Prediction', grid_image, n_total_iter)

            # change lr
            if n_total_iter % 2500 == 0:
                lr_ = lr * 0.1 ** (n_total_iter // 2500)
                for param_group in optimizer_student.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_teacher.param_groups:
                    param_group['lr'] = lr_

            if n_total_iter % n_save_iter == 0:
                # Save model checkpoints
                torch.save(model_student.state_dict(), "%s/model_%d.pth" % (save_path, n_total_iter))
                logging.info("save model : %s/model_%d.pth" % (save_path, n_total_iter))

            if n_total_iter % 10 == 0 and n_total_iter >= 500:
                model_student.eval()
                eval_dice_score = 0
                logging.info('evaluating:')
                for eval_sample in tqdm(eval_dataloader):
                    eval_input = eval_sample['volume'].to('cuda').float()
                    eval_seg = eval_sample['label'].to('cuda').float()
                    eval_seg = eval_seg.cpu().detach().numpy().squeeze()
                    with torch.no_grad():
                        pred_seg = model_student(eval_input)
                        pred_seg = torch.argmax(pred_seg, dim=1)
                        pred_seg = torch.squeeze(pred_seg)
                        pred_seg = pred_seg.cpu().detach().numpy()
                        dice_score_sample = metric.binary.dc(pred_seg, eval_seg)
                        jaccard_score_sample = metric.binary.jc(pred_seg, eval_seg)
                        hd95_score_sample = metric.binary.hd95(pred_seg, eval_seg)
                        asd_score_sample = metric.binary.asd(pred_seg, eval_seg)
                        eval_dice_score += dice_score_sample
                eval_dice_score /= len(eval_dataloader)
                logging.info("evaluation result : %.4f" % eval_dice_score)
                writer.add_scalar('eval_dice_score', eval_dice_score, global_step=n_total_iter)
                model_student.train()

                if eval_dice_score > best_eval_dice:
                    best_eval_iter = n_total_iter
                    best_eval_dice = eval_dice_score
                    torch.save(model_student.state_dict(), "%s/model_best_dice.pth" % save_path)
                    logging.info("saving best model/dice -- iteration number:%d" % best_eval_iter)
                    writer.add_scalar('best_model/dice', best_eval_dice, best_eval_iter)


        pseudo_label_avg_dice /= len(train_dataloader)
        writer.add_scalar('Dice/unlabeled', pseudo_label_avg_dice, n_total_iter)
        student_pred_avg_dice /= len(train_dataloader)
        writer.add_scalar('Dice/labeled', student_pred_avg_dice, n_total_iter)

    torch.save(model_student.state_dict(), "%s/model_%d.pth" % (save_path, n_total_iter))
    logging.info("save model : %s/model_%d.pth" % (save_path, n_total_iter))
    writer.close()


if __name__ == "__main__":
    train(**vars(arg))

