import torch
from hausdorff import hausdorff_distance

SMOOTH = 1e-6

def h_loss(y_true, y_pred, gamma=5, alpha=0.25, sigmoid=0.001):
    y_true = y_true.float()
    alpha_t = y_true * alpha + (torch.ones_like(y_true) - y_true) * (1 - alpha)
    p_t = y_true * y_pred + (torch.ones_like(y_true) - y_true) * (torch.ones_like(y_true) - y_pred) + SMOOTH
    myloss = - 10 * (1 + sigmoid) * alpha_t * torch.pow((torch.ones_like(y_true) - p_t), gamma - 1) * torch.log(p_t) / (
            sigmoid + torch.pow((torch.ones_like(y_true) - p_t), gamma))
    myloss = torch.clamp(myloss, SMOOTH, 1. - SMOOTH)
    return torch.mean(myloss)

def topk_loss(y_true, y_pred, threshold=0.7):
    y_pred = torch.clamp(y_pred, SMOOTH, 1. - SMOOTH)
    ce = y_true * torch.log(y_pred) + (1. - y_true) * torch.log(1. - y_pred)
    weight1 = torch.where(y_pred < threshold, 1., 0.) * y_true
    weight2 = torch.where(y_pred > (1. - threshold), 1. ,0.) * (1. - y_true)
    loss = -(torch.sum(weight1 * ce + weight2 * ce) / (torch.sum(weight1) + torch.sum(weight2) + SMOOTH))
    return torch.mean(loss)

def dice_coef(y_true, y_pred):
    N = y_true.shape[0]
    y_pred_f = y_pred.view(N, -1)
    y_true_f = y_true.view(N, -1)
    intersection = torch.sum(y_true_f * y_pred_f, dim=1)
    union = torch.sum(y_true_f, dim=1) + torch.sum(y_pred_f, dim=1)
    return torch.mean((2. * intersection + SMOOTH) / (union + SMOOTH))

def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def h_dice_loss(y_true, y_pred):
    hloss = h_loss(y_true=y_true, y_pred=y_pred)
    dice = dice_loss(y_true=y_true, y_pred=y_pred)
    print('h_loss: {:5f}  dice_loss: {:5f}'.format(hloss.item(), dice.item()))
    return hloss + dice

def dice_score(y_true, y_pred):
    N = y_true.shape[0]
    y_pred_f = y_pred.view(N, -1).round()
    y_true_f = y_true.view(N, -1)
    intersection = torch.sum(y_true_f * y_pred_f, dim=1)
    union = torch.sum(y_true_f, dim=1) + torch.sum(y_pred_f, dim=1)
    return torch.mean((2. * intersection + SMOOTH) / (union + SMOOTH))

def get_hausdorff_distance(y_true, y_pred):
    hd95 = 0
    for i in range(y_true.shape[0]):
        hd95 += hausdorff_distance(y_pred[i].squeeze(0).round().cpu().numpy(), y_true[i].squeeze(0).cpu().numpy())
    return hd95

def get_evaluation(y_true, y_pred, threshold=0.5):
    N = y_true.shape[0]
    y_pred_f = y_pred.view(N, -1)
    y_true_f = y_true.view(N, -1)

    TP = ((y_pred_f >= threshold) & (y_true_f == 1)).sum(dim=1)
    TN = ((y_pred_f < threshold) & (y_true_f == 0)).sum(dim=1)
    FN = ((y_pred_f < threshold) & (y_true_f == 1)).sum(dim=1)
    FP = ((y_pred_f >= threshold) & (y_true_f == 0)).sum(dim=1)

    dice = (2 * TP + SMOOTH) / (2 * TP + FP + FN + SMOOTH)
    iou = TP / (TP + FN + FP + SMOOTH)
    pre = TP / (TP + FP + SMOOTH)
    hd95 = get_hausdorff_distance(y_true, y_pred)
    return torch.mean(dice), torch.mean(iou), torch.mean(pre), hd95 / N