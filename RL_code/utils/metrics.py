# dc = dice(torch.argmax(outputs_soft[:labeled_bs], dim=1), label_batch[:labeled_bs])
def dice(input, target, ignore_index=None):
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1).float()
    tflat = target.clone().view(-1).float()
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum().float()

    return (2. * intersection + smooth) / (iflat.sum().float() + tflat.sum().float() + smooth)






