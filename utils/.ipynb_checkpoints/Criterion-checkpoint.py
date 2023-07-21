__author__ = 'Titi'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils.laplacian_pyramid_loss import *


class LapLoss(nn.Module):
    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu')):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        
    def forward(self, predImg, target):
        pyr_predImg  = laplacian_pyramid(img=predImg, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_predImg, pyr_target))

class GDL(nn.Module):
    """
    Gradient Difference Loss
    Image gradient difference loss as defined by Mathieu et al. (https://arxiv.org/abs/1511.05440).
    """
    def __init__(self):
        super(GDL, self).__init__()
        self.alpha = 1
        
    def forward(self, predImg, target):
        # [seq_length, channels, height, width]
        predImg_col_grad = torch.abs(predImg[:, :, :, :-1] - predImg[:, :, :, 1:])
        predImg_row_grad = torch.abs(predImg[:, :, 1:, :] - predImg[:, :, :-1, :])
        target_col_grad = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_row_grad = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        col_grad_loss = torch.abs(predImg_col_grad - target_col_grad)
        row_grad_loss = torch.abs(predImg_row_grad - target_row_grad)
        
        #loss = col_grad_loss + row_grad_loss
        loss = torch.sum(col_grad_loss ** self.alpha) + torch.sum(row_grad_loss ** self.alpha)
        return loss

class TV(nn.Module):
    def __init__(self):
        super(TV, self).__init__()
        self.weight = 1
        
    def forward(self, predImg, target):
        # [seq_length, channels, height, width]
        seq_length = predImg.size()[0]
        h_x = predImg.size()[2]
        w_x = predImg.size()[3]
        count_h = (predImg.size()[2] - 1) * predImg.size()[3]
        count_w = predImg.size()[2] * (predImg.size()[3] - 1)
        h_tv = torch.pow((predImg[:, :, 1:, :] - predImg[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((predImg[:, :, :, 1:] - predImg[:, :, :, :w_x-1]), 2).sum()
        
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / seq_length

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        
        # extract conv5 4 features from the VGG-19 model pretrained on ImageNet dataset 
        self.vgg = nn.Sequential(*modules[:35])
        self.vgg = nn.DataParallel(self.vgg).cuda()
        self.vgg.requires_grad = False
        
    def forward(self, predImg, target):
        def _forward(x):
            x = self.vgg(x)
            return x
        # convert to 3 channel
        predImg = predImg.repeat([1, 3, 1, 1]) 
        target = target.repeat([1, 3, 1, 1])
        
        with torch.no_grad():
            vgg_pred = _forward(predImg)
            vgg_target = _forward(target)
        loss = F.mse_loss(vgg_pred, vgg_target)
        
        return loss
        
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-12

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
    
    
class Loss(nn.Module):
    def __init__(self, loss_type):
        super(Loss, self).__init__()
        
        self.loss_function = {}
        if 'L1' in loss_type:
            self.loss_function['L1'] = nn.L1Loss()
        if 'L2' in loss_type:
            self.loss_function['L2'] = nn.MSELoss()
        if 'vgg' in loss_type:
            self.loss_function['vgg'] = VGG()
        if 'char' in loss_type:
            self.loss_function['char'] = L1_Charbonnier_loss()
        if 'gdl' in loss_type:
            self.loss_function['gdl'] = GDL()
        if 'tv' in loss_type:
            self.loss_function['tv'] = TV()
        if 'lap' in loss_type:
            self.loss_function['lap'] = LapLoss(channels=1, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            
        self.loss_type = loss_type
        print("Loss: {}".format(self.loss_function.keys()))
        
        
    def forward(self, pred_tensor, gt_tensor):  
        loss_value = {
            'all_loss': 0
        }

        for l in self.loss_type:
            loss_value[l] = self.loss_function[l](pred_tensor, gt_tensor)
            if l == 'L1' and len(self.loss_type) == 1:
                loss_value[l] *= 1.0
            elif l == 'L1':
                loss_value[l] *= 0.9
            elif l == 'vgg':
                loss_value[l] *= 0.01
            elif l == 'gdl':
                loss_value[l] *= 0.00005
            elif l == 'tv':
                loss_value[l] *= 1
            elif l == 'lap':
                loss_value[l] *= 1
            loss_value['all_loss'] += loss_value[l]
        
        return loss_value
        