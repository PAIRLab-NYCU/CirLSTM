__author__ = 'Titi'

from test_dataset import test_Dataset
from model_factory import Model
from utils.tools import init_meters, init_test_meters, init_loss
from utils.CT_tools import *

import torch
from torch.utils.data import DataLoader

import numpy as np
import os
import argparse
import glob
from tqdm import tqdm
from skimage.measure import compare_ssim
import time
import json
import cv2

from pydicom import dcmread

def process_command():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--patient', help='patient number')
    
    parser.add_argument('--ckpt', help='test checkpoint path')
    parser.add_argument('--test_folder', default='../Full_dose_npy/Valid')
    parser.add_argument('--save_dir', default='../MVFI_output')
    parser.add_argument('--save_results', action="store_true")
    parser.add_argument('--training_name', default='test')
    
    parser.add_argument('-m', '--model_name', default='BiLSTM')
    parser.add_argument('-l', '--seq_length', required=True, type=int)
    parser.add_argument('--h', default=64, type=int)
    parser.add_argument('--w', default=512, type=int)
    parser.add_argument('-c', '--channel', default=1, type=int)
    
    # data setting
    parser.add_argument('--lost_interval', default=4, type=int)
    parser.add_argument('--max', type=int, help='Normalization Max pixel')
    parser.add_argument('--min', type=int, help='Normalization Min pixel')
    
    # model setting
    parser.add_argument('--num_hidden', default='64,64')
    parser.add_argument('--filter_size', default=5, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('-p', '--patch_size', default=4, type=int)
    
    # training setting
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('--loss', default='L1+L2',
                        help='ex. []+[] (L1, L2, vgg)')
    
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--num_gpu', default=2, type=int)
    
    return parser.parse_args()
   
def initialize(args):
    print("< Initialize cuDNN...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Loading LSTM model
    print("< Loading LSTM model...")
    LSTM = Model(args, device)
    print("> Loading Done")

    try:
        print("< Loading [{}] checkpoint...".format(args.ckpt))
        checkpoint = torch.load(args.ckpt)
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        LSTM.load_checkpoint(model_state_dict, optimizer_state_dict)
    except:
        raise "[Error] Loading checkpoint file failed"
        
    return LSTM

def reconstruct(pd_sino_npy, gt_sino_npy, args):
    lost_sino_npy = gt_sino_npy[::args.lost_interval]
    
    pd_gt_lost_sino = {
        'pd': np.array(pd_sino_npy),
        'gt': np.array(gt_sino_npy),
        'lost': np.array(lost_sino_npy)
    }
    
    pd_gt_lost_ct = {}
    
    for k in pd_gt_lost_sino.keys():
        if k == 'lost':
#             continue
            sino = pd_gt_lost_sino[k].reshape(1152//args.lost_interval*args.h, args.w)
            sino_id, sino = create_sino_id(sino, x_max=1152//args.lost_interval)
            pd_gt_lost_ct[k] = reconstruction(sino_id)
            
        else:
#             if k == 'gt':
#                 continue
            sino = pd_gt_lost_sino[k].reshape(1152*args.h, args.w)
            sino_id, sino = create_sino_id(sino, x_max=1152)
            pd_gt_lost_ct[k] = reconstruction(sino_id)

    return pd_gt_lost_ct
    
# predict folder list
def predict(proj_folder, model, args, index):
    
    psnrs, ssims = [], []
    for _ in range(args.seq_length):
        init_psnr, init_ssim = init_meters()
        psnrs.append(init_psnr)
        ssims.append(init_ssim)

    all_prediction = [None for i in range(1152)]
    all_gt = [None for i in range(1152)]

    test_dataset = test_Dataset(proj_folder, index=index, seq_length=args.seq_length,
                                h=args.h, w=args.w, circle_num=1152,
                                lost_interval=args.lost_interval, norm_max=args.max, norm_min=args.min)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    start_time = time.time()
    with torch.no_grad():
        for path, lost_proj, proj, _index_array in tqdm(test_loader):
            psnrs, ssims, pred_seq = model.evaluate(path, lost_proj, proj, psnrs, ssims)

            for p, batch_pred_seq, batch_proj in zip(path, pred_seq, proj):
                start_index = int(p.split(' ')[1])
                indexes = range(start_index, start_index+args.seq_length)
                for i, i_pred_seq, i_proj in zip(indexes, batch_pred_seq, batch_proj):
                    if i >= 1152:
                        break
                    all_prediction[i] = i_pred_seq.cpu().detach().numpy()[0]
                    all_gt[i] = i_proj.numpy()[0]

    # reconstruction
    pd_gt_lost_ct = reconstruct(all_prediction, all_gt, args)
    print("Processing time: {}".format(time.time()-start_time))
    raise
    pd_ct = pd_gt_lost_ct['pd']
    gt_ct = pd_gt_lost_ct['gt']
    lost_ct = pd_gt_lost_ct['lost']
    
#     np.save('gt.npy', gt_ct)
#     np.save('pd.npy', pd_ct)
#     np.save('lost.npy', lost_ct)
    
    ct_max = gt_ct.max()
    if ct_max != 0:
        gt_ct /= ct_max
        pd_ct /= ct_max
        lost_ct /= ct_max
    
    pd_rmse, pd_psnr, pd_ssim = RMSE(gt_ct, pd_ct), calc_psnr(gt_ct, pd_ct, PIXEL_MAX=1), compare_ssim(gt_ct, pd_ct)
    lost_rmse, lost_psnr, lost_ssim = RMSE(gt_ct, lost_ct), calc_psnr(gt_ct, lost_ct, PIXEL_MAX=1), compare_ssim(gt_ct, lost_ct)
    
    print("Predict")
    print("RMSE: {:.03f}, PSNR: {:.03f}, SSIM: {:.03f}".format(pd_rmse, pd_psnr, pd_ssim))
    print("Lost")
    print("RMSE: {:.03f}, PSNR: {:.03f}, SSIM: {:.03f}".format(lost_rmse, lost_psnr, lost_ssim))
    
    return [pd_rmse, pd_psnr, pd_ssim], [lost_rmse, lost_psnr, lost_ssim]

class record_values():
    def __init__(self):
        self.values = []
    def update(self, value):
        self.values.append(value)
    def mean(self):
        return np.mean(self.values)
    def std(self):
        return np.std(self.values)
    
def record(all_slice):
    for k in all_results.keys():
        all_results[k].update(np.mean(all_slice[k]))

def main():
    args = process_command()
    
    slice_paths = glob.glob("../Full_dose_images/Valid/{}_Full_dose_images/*".format(args.patient))
#     slice_paths = glob.glob("../NCKU_images/{}/*".format(args.patient))
    print(slice_paths)
        
    # Initialize model
    model = initialize(args)
    
    slices_length = len(slice_paths)
    all_slice = {
        'PD_RMSE': [None for _ in range(slices_length)],
        'PD_PSNR': [None for _ in range(slices_length)],
        'PD_SSIM': [None for _ in range(slices_length)],
        'LOST_RMSE': [None for _ in range(slices_length)],
        'LOST_PSNR': [None for _ in range(slices_length)],
        'LOST_SSIM': [None for _ in range(slices_length)],
    }
    
    for index, slice_path in enumerate(slice_paths):
        slice_npy_folder = slice_path.replace('images', 'npy')[:-4]
        print(slice_npy_folder)
        pd_metric, lost_metric = predict(slice_npy_folder, model, args, index)
        
        all_slice['PD_RMSE'][index] = pd_metric[0]
        all_slice['PD_PSNR'][index] = pd_metric[1]
        all_slice['PD_SSIM'][index] = pd_metric[2]
        
        all_slice['LOST_RMSE'][index] = lost_metric[0]
        all_slice['LOST_PSNR'][index] = lost_metric[1]
        all_slice['LOST_SSIM'][index] = lost_metric[2]
    
    print("###")
    print("Predict")
    print("RMSE: {:.03f}, PSNR: {:.03f}, SSIM: {:.03f}".format(np.mean(all_slice['PD_RMSE']), np.mean(all_slice['PD_PSNR']), np.mean(all_slice['PD_SSIM'])))
    
    print("Lost")
    print("RMSE: {:.03f}, PSNR: {:.03f}, SSIM: {:.03f}".format(np.mean(all_slice['LOST_RMSE']), np.mean(all_slice['LOST_PSNR']), np.mean(all_slice['LOST_SSIM'])))
    
if __name__ == '__main__':
    main()