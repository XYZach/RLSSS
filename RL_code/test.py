import os
import numpy as np
import argparse
import torch
import json
from RL_code.networks.vnet import VNet
from RL_code.test_util import test_all_case
import logging
import sys
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/raid/zxy/datasets/2018LA_Seg_Training Set', help='Name of Experiment')
parser.add_argument('--list_dir', type=str, default='../data', help='folder of train&test list')
parser.add_argument('--model', type=str, default='LAHeart_RL_pretrain_batch4', help='model_name')  # 跑前修改
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--num_classes', type=int, default=2, help='number of class')
parser.add_argument('--n_filters', type=int, default=16, help='number of filters')
parser.add_argument('--normalization', type=str, default='batchnorm', help='groupnorm or batchnorm')
parser.add_argument('--patch_size', type=tuple, default=(112, 112, 80), help='patch_size')

parser.add_argument('--model_type', type=str, default='student', help='model_type: student, teacher, backbone')
parser.add_argument('--iter_num', type=int, default=6000, help='iter_num')

parser.add_argument('--save_result', type=bool, default=True, help='save_result: True or False')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = os.path.join("../model/", FLAGS.model)
# snapshot_path = os.path.join("/raid/zxy/codes/semi-supervised-seg/UA-MT/model/", FLAGS.model)

n_filters = FLAGS.n_filters
num_classes = FLAGS.num_classes
model_type = FLAGS.model_type
patch_size = FLAGS.patch_size
iter_num = FLAGS.iter_num
save_result = FLAGS.save_result
if model_type == 'teacher':
    pretrained_model_path = os.path.join(snapshot_path, f'iter_{iter_num}_teacher.pth')
elif model_type == 'student':
    pretrained_model_path = os.path.join(snapshot_path, f'iter_{iter_num}_student.pth')
elif model_type == 'backbone':
    pretrained_model_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')

test_save_path = os.path.join("../model/prediction", FLAGS.model)
test_save_path = os.path.join(test_save_path, os.path.basename(os.path.splitext(pretrained_model_path)[0]))
# test_save_path = os.path.join(test_save_path, os.path.basename(os.path.splitext(pretrained_model_path)[0])+'_train_mode')
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
# save this .py
py_path_old = sys.argv[0]
py_path_new = os.path.join(test_save_path, os.path.basename(py_path_old))
shutil.copy(py_path_old, py_path_new)

with open(os.path.join(FLAGS.list_dir, 'test.list'), 'r') as f:
    image_list = f.readlines()
# with open(os.path.join(FLAGS.list_dir, 'test_unlabeled16.list'), 'r') as f:
#     image_list_unlabeled = f.readlines()
# image_list = image_list + image_list_unlabeled
image_list = [os.path.join(FLAGS.root_path, item.replace('\n', ''), "mri_norm2.h5") for item in image_list]


def test_calculate_metric():
    net = VNet(n_channels=1, n_classes=num_classes, n_filters=n_filters, normalization=FLAGS.normalization, has_dropout=False).cuda()
    net.load_state_dict(torch.load(pretrained_model_path))
    print("init weight from {}".format(pretrained_model_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=patch_size, stride_xy=18, stride_z=4,
                               save_result=save_result, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    logging.basicConfig(filename=os.path.join(test_save_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    metric = test_calculate_metric()
    log_line = 'init weight from:{}\nDice:{:.4f} Jaccard:{:.4f} ASD:{:.4f} HD95:{:.4f}'\
        .format(pretrained_model_path, metric[0], metric[1], metric[2], metric[3])
    logging.info(log_line)

