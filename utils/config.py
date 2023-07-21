__author__ = 'Titi'

import argparse

def process_command():
    parser = argparse.ArgumentParser()
    
    # data I/O
    parser.add_argument('--dataset_name', default='base_dataset',
                        help='The name of dataset.')
    parser.add_argument('--train_data_paths', default='videolist/UCF-101/train_data_list.txt',
                        help='train data paths.')
    parser.add_argument('--valid_data_paths', default='videolist/UCF-101/val_data_list_sample320.txt',
                        help='validation data paths.')
    parser.add_argument('--save_dir', default='../MVFI_output')
    parser.add_argument('--training_name', default='test')
    parser.add_argument('--save_results', action="store_true",
                        help='Is save the predictions')
    
    # model
    parser.add_argument('--model_name',
                        required=True,
                        help='Model name')
    parser.add_argument('--seq_length',
                        required=True,
                        type=int,
                        help='sequence length') 
    parser.add_argument('--img_size',
                        required=True,
                        type=int,
                        help='image size')
    parser.add_argument('--img_channel',
                        required=True,
                        type=int,
                        help='image channel')
    parser.add_argument('--num_hidden', default='64,64,64,64',
                        help='Hidden state channel')
    parser.add_argument('--filter_size', default=5,
                        type=int,
                        help='The filter size of convolution in the lstm')
    parser.add_argument('--stride', default=1,
                        type=int,
                        help='The stride of convolution in the lstm')
    parser.add_argument('--patch_size', default=4,
                        type=int,
                        help='PixelShuffle parameter')
    parser.add_argument('--LSTM_pretrained', default='',
                        help="LSTM pretrained model path")
    parser.add_argument('--CA_patch_size', default=8,
                        type=int,
                        help="Channel attention pixelshuffle factor")
    parser.add_argument('--n_resgroups', default=5,
                        type=int,
                        help="Channel attention number of Residual Groups")
    parser.add_argument('--n_resblocks', default=12,
                        type=int,
                        help="Channel attention number of Residual Blocks")
    
    # training setting
    parser.add_argument('--lr', default=0.001,
                        type=float,
                        help='base learning rate')
    parser.add_argument('--batch_size', default=4,
                        type=int,
                        help='batch size for training.')
    parser.add_argument('--epochs', default=100,
                        type=int,
                        help='batch size for training.')
    parser.add_argument('--delta', default=0.1,
                        type=float,
                        help='teacher ratio of mask probability')
    parser.add_argument('--checkpoint_path', default='',
                        help='checkpoint path')
    parser.add_argument('--loss', default='L1+L2',
                        help='ex. []+[] (L1, L2, vgg)')
    parser.add_argument('--num_workers', default=12,
                        type=int,
                        help='data loader num_workers')
    
    return parser.parse_args()