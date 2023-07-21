__author__ = 'Titi'

from utils.pytorch_msssim import ssim_matlab as ssim_pth

import numpy as np
import math
import torch
import cv2

def seq_pixel_shuffle(input, scale_factor):
    batch_size, seq_length, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, seq_length, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 2, 5, 3, 6, 4).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, seq_length, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 2, 4, 6, 3, 5).contiguous()

    return shuffle_out.view(batch_size, seq_length, out_channels, out_height, out_width)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def init_meters():
    rmses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return rmses, psnrs, ssims

def init_test_meters():
    RMSE = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return RMSE, psnrs, ssims

def init_loss(loss_type):
    loss_dict = {
        'all_loss': AverageMeter()
    }
    for l_type in loss_type:
        loss_dict[l_type] = AverageMeter()
    return loss_dict

def calc_psnr(img1, img2):
    '''
        Here we assume quantized(0-255) arguments.
    '''
    diff = (img1 - img2).div(1)
    mse = diff.pow(2).mean() + 1e-8
    
    return -10 * math.log10(mse)

def quantize(img, rgb_range=255):
    return img.mul(255 / rgb_range).clamp(0, 255).round()
    
def calc_metrics(img1, img2):
#     q_img1 = quantize(img1, rgb_range=1.)
#     q_img2 = quantize(img2, rgb_range=1.)
    
    psnr = calc_psnr(img1, img2)
    ssim = ssim_pth(img1.unsqueeze(0), img2.unsqueeze(0), val_range=1)
    
    return psnr, ssim, img1, img2

def save_image(img, path, color_mode='RGB'):
    img = img[0].cpu().detach().numpy()
    if color_mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif color_mode == 'BGR' or color_mode == 'GRAY':
        pass
    else:
        raise
    np.save(path, img)