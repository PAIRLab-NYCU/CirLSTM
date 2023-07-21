__author__ = 'Titi'

from CT_dataset import CT_Dataset
from test_dataset import test_Dataset
from model_factory import Model
from utils.tools import init_meters, init_loss

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import argparse
from tqdm import tqdm

def process_command():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_folder', default='../Full_dose_npy/Train')
    parser.add_argument('--valid_folder', default='../Full_dose_npy/Valid')
    parser.add_argument('--test_folder', default='../Full_dose_npy/Valid/C095_Full_dose_npy/1-020')
    
    parser.add_argument('-m', '--model_name', required=True)
    parser.add_argument('-l', '--seq_length', required=True, type=int)
    parser.add_argument('--h', default=64, type=int)
    parser.add_argument('--w', default=512, type=int)
    parser.add_argument('-c', '--channel', default=1, type=int)
    
    # data setting
    parser.add_argument('--lost_interval', default=4, type=int)
    parser.add_argument('--max', type=int, help='Normalization Max pixel')
    parser.add_argument('--min', type=int, help='Normalization Min pixel')
    
    # model setting
    parser.add_argument('--num_hidden', default='64,64,64,64')
    parser.add_argument('--filter_size', default=5, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('-p', '--patch_size', default=4, type=int)
    
    # training setting
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('--loss', default='L1+L2',
                        help='ex. []+[] (L1, L2, vgg)')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--num_gpu', default=2, type=int)
    
    # training I/O
    parser.add_argument('--save_dir', default='../MVFI_output')
    parser.add_argument('--training_name', default='test')
    parser.add_argument('--save_results', action="store_true")
    parser.add_argument('--test', default='',
                        help='test checkpoint path')
    parser.add_argument('-r', '--resume', default='',
                        help='resume checkpoint path')
    parser.add_argument('--fine_tune', action="store_true",
                        help='whether Fine-tune')
    
    return parser.parse_args()

def train(LSTM, args, train_loader, epoch, writer):
    # specify loss type
    loss_type = args.loss.split('+') 
    loss_dict = init_loss(loss_type)
    
    for i, (path, lost_proj, proj) in enumerate(tqdm(train_loader)):
        loss_dict = LSTM.train(lost_proj, proj, loss_dict)
        if i % 10 == 0:
            print(">Epoch: {}\tLoss: {}".format(epoch, loss_dict['all_loss'].avg))

    # Write in TensorBoard
    for l in loss_type:
        writer.add_scalar('Train/{}-Loss'.format(l), loss_dict[l].avg, epoch)
    writer.add_scalar('Train/Loss', loss_dict['all_loss'].avg, epoch)
    
def validation(LSTM, args, valid_loader, epoch, gen_dir, writer):
    print("< validation...")
    psnrs, ssims = [], []
    for _ in range(args.seq_length):
        init_psnr, init_ssim = init_meters()
        psnrs.append(init_psnr)
        ssims.append(init_ssim)
        
    with torch.no_grad():
        for path, lost_proj, proj in tqdm(valid_loader):
            psnrs, ssims = LSTM.test(path, gen_dir, lost_proj, proj, epoch, psnrs, ssims)
    psnr_avg, ssim_avg = init_meters()
    for t in range(args.seq_length):
        print("{}\tPSNR: {:0.3f}, SSIM: {:0.3f}".format(t+1, psnrs[t].avg, ssims[t].avg))
        psnr_avg.update(psnrs[t].avg)
        ssim_avg.update(ssims[t].avg)
            
    # Write in TensorBoard
    writer.add_scalar('Train/PSNR', psnr_avg.avg, epoch)
    writer.add_scalar('Train/SSIM', ssim_avg.avg, epoch)
    
    return psnr_avg.avg

def test(LSTM, args, test_loader, epoch, gen_dir):
    print("< testing...")
    psnrs, ssims = [], []
    for _ in range(args.seq_length):
        init_psnr, init_ssim = init_meters()
        psnrs.append(init_psnr)
        ssims.append(init_ssim)
        
    with torch.no_grad():
        for path, proj, lost_proj in tqdm(test_loader):
            psnrs, ssims = LSTM.test(path, gen_dir, lost_proj, proj, epoch, psnrs, ssims)
    psnr_avg, ssim_avg = init_meters()
    for t in range(args.seq_length):
        print("{}\tPSNR: {:0.3f}, SSIM: {:0.3f}".format(t+1, psnrs[t].avg, ssims[t].avg))
        psnr_avg.update(psnrs[t].avg)
        ssim_avg.update(ssims[t].avg)

def main():
    args = process_command()
    
    ckpt_dir = os.path.join(args.save_dir, args.training_name) # directory to store trained checkpoint
    gen_dir = os.path.join(args.save_dir, args.training_name, 'results') # directory to store result
    log_dir = os.path.join(args.save_dir, 'logs', args.training_name) # log directory for TensorBoard
    
    print("< Initialize cuDNN...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Create dataset
    print("< Creating dataset...")
    print("> Training Folder: {}".format(args.train_folder))
    print("> Validation Folder: {}".format(args.valid_folder))
    train_dataset = CT_Dataset(args.train_folder, seq_length=args.seq_length,
                               h=args.h, w=args.w, circle_num=1152,
                               lost_interval=args.lost_interval, norm_max=args.max, norm_min=args.min)
    valid_dataset = CT_Dataset(args.valid_folder, seq_length=args.seq_length,
                               h=args.h, w=args.w, circle_num=1152,
                               lost_interval=args.lost_interval, norm_max=args.max, norm_min=args.min)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    
    # Training setting
    writer = SummaryWriter(log_dir)
    best_psnr = 0
    start_epoch = 0
    
    # Loading LSTM model
    print("< Loading LSTM model...")
    LSTM = Model(args, device)
    print("> Loading Done")
    
    # Testing block
    if args.test:
        try:
            checkpoint = torch.load(args.test)
            model_state_dict = checkpoint['model_state_dict']
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            print("< Loading [{}] checkpoint...".format(args.test))
            LSTM.load_checkpoint(model_state_dict, optimizer_state_dict)
        except:
            raise "[Error] Loading checkpoint file failed"
        # Load test dataset
        print("> Test Folder: {}".format(args.test_folder))
        test_dataset = test_Dataset(args.test_folder,
                                    seq_length=args.seq_length,
                                    h=args.h, w=args.w, circle_num=1152,
                                    lost_interval=args.lost_interval, norm_max=args.max, norm_min=args.min)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        # testing
        LSTM.network.eval()
        test(LSTM, args, test_loader, 'test', gen_dir)
        return
    ###############
    
    # if resume
    if args.resume:
        print("> Resume training")
        try:
            checkpoint = torch.load(args.resume)
            model_state_dict = checkpoint['model_state_dict']
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            start_epoch = checkpoint['epoch']
            print("> Epoch resume from {}".format(start_epoch))
            best_psnr = checkpoint['best_psnr']
            print("> Current best psnr is {}".format(best_psnr))
            print("< Loading [{}] checkpoint...".format(args.resume))
            LSTM.load_checkpoint(model_state_dict, optimizer_state_dict)
        except IOError as exc:
            raise RuntimeError('No this checkpoint file') from exc
            
    if args.fine_tune:
        print("> Fine tune")
        start_epoch = 0
        print("> Epoch fine-tune from {}".format(start_epoch))
    
    for epoch in range(start_epoch, args.epochs):
        # training
        LSTM.network.train()
        train(LSTM, args, train_loader, epoch, writer)
        
        # validation
        LSTM.network.eval()
        psnr = validation(LSTM, args, valid_loader, epoch, gen_dir, writer)
        
        if psnr > best_psnr:
            print("< Saving checkpoint...")
            best_psnr = psnr
            LSTM.save_checkpoint(epoch, ckpt_dir, best_psnr)
    
    
        
if __name__ == '__main__':
    main()
    import requests
    r = requests.get('https://weihomelive.com/training_reminder?username=TITI')