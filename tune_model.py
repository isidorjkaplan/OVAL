import argparse
import os
import subprocess

# Hyperparameter tuning script for offline model training

if __name__ == "__main__":
    # python3 tune_model.py ARGS 
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning Arguments')

    ##### TRAINING PARAMETERS ######
    # These set various training HPs which optimize offline learning
    # Learning rate
    parser.add_argument('--lr_params', type=list, default=None, help='Learning rate range with (min, max, step)', required=True)
    # Batch size
    parser.add_argument('--bs_params', type=list, default=None, help='Batch size range with (min, max, step)', required=True)
    # Choice of loss function
    parser.add_argument('--loss_function', type=str, default='mse', help='Loss during training (mae, mse, bce)')
    # Number of training epochs per test run
    parser.add_argument('--num_epochs', type=int, default=10, help='Epochs during training')
    # In case we have access to CUDA
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda')

    ##### MODEL PARAMETERS ######
    # These actually change the model.py file to change the model architecture
    # Encoder Convolution Layer 1
    parser.add_argument("--l1_out_channels_params", type=list, default=[5,5,1], help='Out channels in first encoder convolution layer: (min, step, max)')
    parser.add_argument("--l1_stride_params", type=int, default=[2,2,1], help='Stride in first conv layer: (min, max, step')
    parser.add_argument("--l1_kernel_params", type=int, default=[2,2,1], help='Kernel size in first conv layer: (min, max, step)')
    # Encoder Convolution Layer 2
    parser.add_argument("--l2_out_channels_params", type=list, default=[2,2,1], help='Out channels in second encoder convolution layer: (min, step, max)')
    parser.add_argument("--l2_stride_params", type=int, default=[2,2,1], help='Stride in second conv layer: (min, max, step')
    parser.add_argument("--l2_kernel_params", type=int, default=[2,2,1], help='Kernel size in second conv layer: (min, max, step)')

    args = parser.parse_args()

    ##### CREATE PARAMETER VECTORS ######
    epsilon = 1e-9
    args.lr_params[1] += epsilon
    args.bs_params[1] += epsilon
    args.l1_out_channels_params[1] += epsilon
    args.l2_out_channels_params[1] += epsilon
    args.l1_stride_params[1] += epsilon
    args.l2_stride_params[1] += epsilon
    args.l1_kernel_params[1] += epsilon
    args.l2_kernel_params[1] += epsilon
   
    learning_rate = range(*args.lr_params)
    batch_size = range(*args.bs_params)
    l1_out_channels_params = range(*args.l1_out_channels_params)
    l2_out_channels_params = range(*args.l2_out_channels_params)
    l1_stride_params = range(*args.l1_stride_params)
    l2_stride_params = range(*args.l2_stride_params)
    l1_kernel_params = range(*args.l1_kernel_params)
    l2_kernel_params = range(*args.l2_kernel_params)

    ##### RUN THE OFFLINE TRAINING, SAVING THE DATA ######
    # At the moment ignoring model params since we haven't finished the architecture yet
    # Also, make sure to put the training data in data/videos as we've been doing thus far
    for lr in learning_rate:
        for bs in batch_size:
            cmd = f"python3 offline.py --lr {lr} --batch-size {bs} --epochs {args.num_epochs}"
            if args.loss_function:
                cmd += f" --loss {args.loss_function}"
            if args.cuda:
                cmd += f" --cuda"
            os.system(cmd)  