import torch
from torch.nn import functional as F
import numpy as np


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - dice
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def mask_to_one_hot(mask, n_classes):
    """
    Convert a segmentation mask to one-hot coded tensor
    :param mask: mask tensor of size Bx1xDxMxN
    :param n_classes: number of classes
    :return: one_hot: BxCxDxMxN
    """
    one_hot_shape = list(mask.shape)
    one_hot_shape[1] = n_classes

    mask_one_hot = torch.zeros(one_hot_shape).to(mask.device)

    mask_one_hot.scatter_(1, mask.long(), 1)

    return mask_one_hot


def multi_dice_loss(source, target, n_class=2, no_bg=True, weight_type='Simple'):
    assert source.shape[0] == target.shape[0]
    assert source.shape[-3:] == target.squeeze().shape[-3:]
    eps = 1e-5

    shape = list(source.shape)

    # flat the spatial dimensions
    source_flat = source.view(shape[0], shape[1], -1)

    # flat the spatial dimensions and transform it into one-hot coding
    if len(target.shape) == len(shape) - 1:
        target_flat = mask_to_one_hot(target.view(shape[0], 1, -1), n_class)
    elif target.shape[1] == shape[1]:
        target_flat = target.view(shape[0], shape[1], -1)
    else:
        target_flat = None
        raise ValueError("Incorrect size of target tensor: {}, should be {} or []".format(target.shape, shape,
                                                                                          shape[:1] + [1, ] + shape[
                                                                                                              2:]))
    # does not consider background
    if no_bg:
        source_flat = source_flat[:, 1:, :]
        target_flat = target_flat[:, 1:, :]

    #
    source_volume = source_flat.sum(2)
    target_volume = target_flat.sum(2)

    # if weight_type == 'Simple':
    #     # weights = (target_volume.float().sqrt() + self.eps).reciprocal()
    #     # weights = (target_volume.float() ** (1. / 3.) + eps).reciprocal()
    #     weights = (target_volume.float() ** 2 + eps).reciprocal()
    #     # temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
    #     # max_weights = temp_weights.max(dim=1, keepdim=True)[0]
    #     # weights = torch.where(torch.isinf(weights), torch.ones_like(weights)*max_weights, weights)
    # elif weight_type == 'Volume':
    #     weights = (target_volume + eps).float().reciprocal()
    #     # weights = 1/(target_volume ** 2+self.eps)
    #     temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
    #     max_weights = temp_weights.max(dim=1, keepdim=True)[0]  # 为什么第0个是max？
    #     weights = torch.where(torch.isinf(weights), torch.ones_like(weights) * max_weights, weights)
    # elif weight_type == 'Uniform':
    #     weights = torch.ones(shape[0], shape[1] - int(no_bg))
    # else:
    #     raise ValueError("Class weighting type {} does not exists!".format(weight_type))
    # weights = weights / weights.max()
    # # print(weights)
    # weights = weights.to(source.device)

    intersection = (source_flat * target_flat).sum(2)
    scores = (2. * (intersection.float()) + eps) / (
            (source_volume.float() + target_volume.float()) + 2 * eps)

    # return 1 - (weights * scores).sum() / weights.sum()
    return 1 - scores.mean()


def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


if __name__ == '__main__':
    pass

