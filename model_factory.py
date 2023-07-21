__author__ = 'Titi'

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import shutil
import time

from torch.optim import Adam
from torch.nn import DataParallel

import model_all as model
from utils import Criterion
from utils.tools import calc_metrics, save_image, seq_pixel_shuffle
from utils.CT_tools import RMSE

class Model(object):
    def __init__(self, args, device):
        
        self.seq_length = args.seq_length
        print("> Sequence Length: {}".format(self.seq_length))
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.save_results = args.save_results
        
        num_gpu = [i for i in range(args.num_gpu)]
        print("> Use GPUs: {}".format(num_gpu))
        networks_map = {
            'BiLSTM': model.RNN,
        }

        # check model name
        if args.model_name in networks_map:
            Network = networks_map[args.model_name]
            self.network = Network(args)
            self.network = DataParallel(self.network, device_ids=num_gpu)
            self.network.to(device)
        else:
            raise ValueError('Name of network unknown {}'.format(args.model_name))

        self.optimizer = Adam(self.network.parameters(), lr=args.lr)
        
        # specify loss type
        loss_type = args.loss.split('+')
        self.criterion = Criterion.Loss(loss_type)
        self.pred_loss = 0
            
    def train(self, input_tensor, gt_tensor, loss_dict):
        patch_tensor = input_tensor.type(torch.cuda.FloatTensor)
        if self.patch_size > 1:
            patch_tensor = seq_pixel_shuffle(input_tensor, 1 / self.patch_size).type(torch.cuda.FloatTensor)
            origin_patch_tensor = seq_pixel_shuffle(gt_tensor, 1 / self.patch_size).type(torch.cuda.FloatTensor)
        patch_rev_tensor = torch.flip(patch_tensor, (1, ))

        self.optimizer.zero_grad()
        pred_seq = self.network(patch_tensor, patch_rev_tensor, origin_patch_tensor)

        all_loss = 0
        for b in range(pred_seq.shape[0]):
            loss_value = self.criterion(pred_seq[b], gt_tensor[b].type(torch.cuda.FloatTensor))
            all_loss += loss_value['all_loss']
            for key in loss_value:
                loss_dict[key].update(loss_value[key].detach().cpu().numpy())
        all_loss /= self.batch_size
        all_loss.backward()
        
        self.optimizer.step()
        
        return loss_dict
        
    def test(self, vid_path, gen_frm_dir, input_tensor, gt_tensor, epoch, psnrs, ssims):
        gt_tensor = gt_tensor.type(torch.cuda.FloatTensor)
        patch_tensor = seq_pixel_shuffle(input_tensor, 1 / self.patch_size).type(torch.cuda.FloatTensor)
        origin_patch_tensor = seq_pixel_shuffle(gt_tensor, 1 / self.patch_size).type(torch.cuda.FloatTensor)
        patch_rev_tensor = torch.flip(patch_tensor, (1, ))
        
        pred_seq = self.network(patch_tensor, patch_rev_tensor, origin_patch_tensor)

        for batch in range(pred_seq.shape[0]):
            # get file path and name
            try:
                path = vid_path[batch]
            except:
                continue
            f_name = path.split('/')[-1]
            num = 0
            splited = f_name.split(' ')
            dicom_num = splited[-1]
            if len(splited) == 2:
                num = int(splited[-1])
            ep_folder = os.path.join(gen_frm_dir, str(epoch))
            f_folder = os.path.join(ep_folder, dicom_num)
            
            batch_pred_seq = pred_seq[batch]
            batch_gt_seq = gt_tensor[batch]
            
            for t in range(self.seq_length):
                pred_img = batch_pred_seq[t]
                gt_img = batch_gt_seq[t]
                
                psnr, ssim, pred_img, gt_img = calc_metrics(pred_img, gt_img) # (pred, GT)
                
                psnrs[t].update(psnr)
                ssims[t].update(ssim)
                
                # save prediction and GT
                if self.save_results:
                    if not os.path.isdir(f_folder):
                        os.makedirs(f_folder)
                    pred_path = os.path.join(f_folder, "pd-{}.npy".format(num+t+1))
                    save_image(pred_img, pred_path, 'GRAY')
#                     gt_path = os.path.join(f_folder, "gt-{}.npy".format(num+t+1))
#                     save_image(gt_img, gt_path, 'GRAY')
                
        return psnrs, ssims

    
    def save_checkpoint(self, epoch, save_dir, best_psnr):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, 'checkpoint_best.tar')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': best_psnr
        }, save_path)
        
    def load_checkpoint(self, model_state_dict, optimizer_state_dict):
        self.network.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        
    def save_model(self, epoch, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, 'model_{}.pt'.format(epoch))
        torch.save(self.network.module.state_dict(), save_path)
        
    def evaluate(self, vid_path, input_tensor, gt_tensor, rmses, psnrs, ssims):
        gt_tensor = gt_tensor.type(torch.cuda.FloatTensor)
        patch_tensor = seq_pixel_shuffle(input_tensor, 1 / self.patch_size).type(torch.cuda.FloatTensor)
        origin_patch_tensor = seq_pixel_shuffle(gt_tensor, 1 / self.patch_size).type(torch.cuda.FloatTensor)
        patch_rev_tensor = torch.flip(patch_tensor, (1, ))
        
        pred_seq = self.network(patch_tensor, patch_rev_tensor, origin_patch_tensor)

        for batch in range(pred_seq.shape[0]):
            # get file path and name
            try:
                path = vid_path[batch]
            except:
                continue
            
            batch_pred_seq = pred_seq[batch]
            batch_gt_seq = gt_tensor[batch]
            
            for t in range(self.seq_length):
                pred_img = batch_pred_seq[t]
                gt_img = batch_gt_seq[t]
                
                psnr, ssim, pred_img, gt_img = calc_metrics(pred_img, gt_img) # (pred, GT)
                rmse = RMSE(gt_img.cpu().numpy(), pred_img.cpu().numpy())
                
                rmses[t].update(rmse)
                psnrs[t].update(psnr)
                ssims[t].update(ssim)
                
        return rmses, psnrs, ssims, pred_seq