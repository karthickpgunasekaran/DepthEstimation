import torch
import torch.nn as nn
import torch.nn.functional as F
import ssim
from torch.autograd import Variable

from skimage import measure

def initSSim():
    model = ssim.SSIM(window_size=10, size_average=True)
    #print("model",model)
    return model

def SSIM(img1,img2,ssim):
    #print(ssim)
    return ssim(img1,img2)

def gradient(pred):
    D_dy = pred[:, :, 1:] - pred[:, :, :-1]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

def smooth_loss(pred_map):
    lambda_val = 0.001
    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.
    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
        weight /= 2.3  # don't ask me why it works better
    return loss*lambda_val

def calculateLoss(img, recon_img,ssim):
    alpha =0.3
    ssim = alpha * (1-SSIM(img,recon_img,ssim))/2
    l1 =  (img - recon_img)
    l1 = (1-alpha)* torch.mean(torch.abs(l1))
    #print("l1:",l1.requires_grad)
    #print("ssim:",ssim.requires_grad)
    loss = l1+ssim

    return loss,ssim.item(),l1.item()

