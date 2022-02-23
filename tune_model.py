import argparse

# Hyperparameter tuning script for offline model training

if __name__ == "__main__":
    # python3 tune_model.py ARGS 
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning Arguments')

    ##### TRAINING PARAMETERS ######
    # These set various training HPs which optimize offline learning
    # Learning rate
    parser.add_argument('--min_lr', type=float, default=0.001, help='The minimum LR')
    parser.add_argument('--max_lr', type=float, default=0.01, help='The maximum LR')
    parser.add_argument('--num_lr_step', type=int, default=2, help='The number of steps between min and max LR')
    # Batch size
    parser.add_argument('--min_bs', type=int, default=2, help='The minimum LR')
    parser.add_argument('--max_bs', type=int, default=64, help='The maximum LR')
    parser.add_argument('--num_bs_step', type=int, default=2, help='The number of LOGARITHMIC steps between min and max LR')
    # Choice of loss function
    parser.add_argument('--loss_function', type=str, default='mse', help='Loss during training (mae, mse, bce)')
    # Number of training epochs per test run
    parser.add_argument('--num_epochs', type=int, default=10, help='Epochs during training')
    # In case we have access to CUDA
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda')
    
    ##### MODEL PARAMETERS ######
    # These actually change the model.py file to change the model architecture
    # Encoder Convolution Layer 1
    parser.add_argument("--l1_out_channels", type=int, default=5, help='Out channels in first encoder convolution layer')
    parser.add_argument("--l1_stride", type=int, default=2, help='Stride in first conv layer')
    parser.add_argument("--l1_kernel", type=int, default=2, help='Kernel size in first conv layer')
    # Encoder Convolution Layer 2
    parser.add_argument("--l2_out_channels", type=int, default=5, help='Out channels in second encoder convolution layer')
    parser.add_argument("--l2_stride", type=int, default=2, help='Stride in second conv layer')
    parser.add_argument("--l2_kernel", type=int, default=2, help='Kernel size in second conv layer')