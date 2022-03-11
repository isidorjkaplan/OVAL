import argparse
import os
import subprocess
import numpy as np
import time
import math

# Hyperparameter tuning script for offline model training

if __name__ == "__main__":
    # python3 online_runner.py ARGS 
    parser = argparse.ArgumentParser(description='Online Training Arguments')
    parser.add_argument('--test_folder', type=str, default=None, help='Test video folder', required=True)
    parser.add_argument('--time_bound', type=int, default=60, help='Number of minutes after which to stop online learning')
    parser.add_argument('--offline_model', type=str, default=None, help='Path to the saved offline model', required=True)

    args = parser.parse_args()

    ##### RUN THE ONLINE TRAINING, SAVING THE DATA ######
    fldr = f"./test_online_{math.ceil(time.time())}"
    os.system(f"mkdir {fldr}")
    start = os.times()[0]

    for f in args.test_folder: # for every test video
        cmd = f"python3 online.py --video {f} --load_model {args.offline_model}"
        cmd += f" --save_model {fldr}/lr_{lr}-bs_{bs}/model.pt"
        if args.loss_function:
            cmd += f" --loss {args.loss_function}"
        if args.cuda:
            cmd += f" --cuda"
        os.system(cmd)
    