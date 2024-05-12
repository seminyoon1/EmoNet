#!/usr/local/bin/ipython
import os
import argparse
from data_loader import get_loader
from torch.backends import cudnn
import torch
import glob
import math
import ipdb
import imageio
import numpy as np
import config as cfg
from solver import Solver    
from models.resnet50 import Classifier  # Import your model class here

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.result_save_path):
        os.makedirs(config.result_save_path)    

    img_size = config.image_size

    # Initialize your model
    model = Classifier()  # Update this with your model class and appropriate initialization parameters
  
    # Load the pre-trained weights if specified
    if config.pretrained_model:
        print(config.pretrained_model)
        pretrained_weights = torch.load(config.pretrained_model)
        model.load_state_dict(pretrained_weights)
        print("Pre-trained weights loaded successfully.")
    else:
        print("Warning: Pre-trained weights not specified. Training from scratch.")

    # Prepare data loader
    data_loader = get_loader(metadata_path=config.metadata_path,
                             image_size=img_size,
                             batch_size=config.batch_size,
                             mode='train',
                             fold=config.fold,
                             num_workers=config.num_workers)
    
    solver = Solver(data_loader, config)
      
    if config.DISPLAY_NET: 
        solver.display_net()
        return
    
    if config.mode == 'train':

        best_accuracy = 0.5058 # Initialize best accuracy
    # For loop for training epochs
        for epoch in range(config.num_epochs):
            print(f"Epoch [{epoch+1}/{config.num_epochs}]")

            # Training phase
            solver.train()

            # Validation phase
            current_accuracy = solver.val(load=True)[0] * 0.01
            

            # Check if current accuracy is higher than the best accuracy
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy

                # Save model checkpoint
                model_path = os.path.join(config.model_save_path, f"best_model.pth")
                torch.save(solver.C.state_dict(), model_path)
                print(f"Saved best model checkpoint: {model_path}")
            else:
                print("Accuracy not improved. Skipping saving the model.")
                
    elif config.mode == 'test':
        solver.val(load=True, plot=True)
    elif config.mode == 'sample':
        solver.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.0001)

    # Training settings
    parser.add_argument('--dataset', type=str, default='EmotionNet', choices=['EmotionNet'])
    parser.add_argument('--num_epochs', type=int, default=99)
    parser.add_argument('--num_epochs_decay', type=int, default=100)
    parser.add_argument('--stop_training', type=int, default=5) #How many epochs after acc_val is not increasing
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--BLUR', action='store_true', default=False) 
    parser.add_argument('--GRAY', action='store_true',  default=False) 
    parser.add_argument('--DISPLAY_NET', action='store_true', default=False) 

    # Test settings
    parser.add_argument('--test_model', type=str, default='')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample'])
    parser.add_argument('--use_tensorboard', action='store_true', default=False)
    parser.add_argument('--GPU', type=str, default='3')

    # Path
    parser.add_argument('--metadata_path', type=str, default='./data/dataset/Dog Emotion')
    parser.add_argument('--log_path', type=str, default='./snapshot/logs')
    parser.add_argument('--model_save_path', type=str, default='./snapshot/models') 
    parser.add_argument('--result_save_path', type=str, default='./snapshot/results') 
    parser.add_argument('--fold', type=str, default='0', choices=['0', '1', '2', 'all'])
    parser.add_argument('--mode_data', type=str, default='normal', choices=['normal', 'aligned'])  

    parser.add_argument('--finetuning', type=str, default='Imagenet', choices=['Imagenet', 'RANDOM'])   
    parser.add_argument('--pretrained_model', type=str, default='./snapshot/models/EmotionNet/normal/fold_0/Imagenet/10_85.pth')    
    # Step size
    parser.add_argument('--log_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=20000)

    config = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    config = cfg.update_config(config)

    print(config)
    main(config)

    ## python main.py --image_size 320 --lr 0.001 --num_epochs 5 --batch_size 100 --fold 0 --mode train
    ## python main.py --image_size 320 --lr 0.001 --num_epochs 1 --batch_size 100 --fold 0 --mode test