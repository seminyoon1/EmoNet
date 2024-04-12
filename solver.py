import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import tqdm
import torchvision.models as models
from PIL import Image
import time
import datetime
from models.resnet50 import Classifier
import ipdb
import config as cfg
from config import num_classes
import glob
import pylab
import pickle
from utils import plot_confusion_matrix
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from graphviz import Digraph
from torchviz import make_dot, make_dot_from_trace
from utils import pdf2png


import warnings
warnings.filterwarnings('ignore')

class Solver(object):

  def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.LOSS = nn.CrossEntropyLoss()

        self.image_size = config.image_size
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.pretrained_model = config.pretrained_model
        self.use_tensorboard = config.use_tensorboard
        self.finetuning = config.finetuning
        self.stop_training = config.stop_training
        self.BLUR = config.BLUR
        self.GRAY = config.GRAY
        self.DISPLAY_NET = config.DISPLAY_NET
        self.loss_fn = nn.CrossEntropyLoss()

        self.test_model = config.test_model
        self.metadata_path = config.metadata_path

        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.result_save_path = config.result_save_path
        self.fold = config.fold
        self.mode_data = config.mode_data

        self.log_step = config.log_step
        self.GPU = config.GPU

        # Build tensorboard if use
        if config.mode!='sample':
          self.build_model()
          if self.use_tensorboard:
            self.build_tensorboard()

          # Start with trained model
          if self.pretrained_model:
            self.load_pretrained_model()
  
  def ACC_TEST(solver, data_loader, mode='VAL', verbose=False):
    # Initialize variables to track accuracy and loss
    correct = 0
    total = 0
    loss_total = 0.0

    # Set model to evaluation mode
    solver.C.eval()

    # Iterate over the data loader
    for i, (images, labels) in enumerate(data_loader):
        # Forward pass to get outputs
        outputs = solver.C(images)

        # Compute loss (if needed)
        loss = solver.loss_fn(outputs, labels)
        loss_total += loss.item()

        # Get predicted labels
        _, predicted = torch.max(outputs, 1)

        # Compute accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Compute accuracy and loss
    accuracy = 100 * correct / total
    average_loss = loss_total / len(data_loader)

    # Print verbose information if requested
    if verbose:
        print(f'{mode} Accuracy: {accuracy:.2f}%')
        print(f'{mode} Loss: {average_loss:.4f}')

    print(loss_total)

    return accuracy, accuracy, average_loss

  #=======================================================================================#
  #=======================================================================================#
  def display_net(self):
	  y = self.C(self.to_var(torch.randn(1,3,224,224)))
	  g = make_dot(y, params=dict(self.C.named_parameters()))
	  filename='network'
	  g.filename=filename
	  g.render()
	  os.remove(filename)
	  pdf2png(filename)
	  print('Network saved at {}.png'.format(filename))

  #=======================================================================================#
  #=======================================================================================#
  def build_model(self):
        # Define your classifier model
        self.C = Classifier(num_classes=self.num_classes)

        # Load pre-trained weights if specified
        if self.pretrained_model:
            pretrained_weights = torch.load(self.pretrained_model)
            self.C.load_state_dict(pretrained_weights)
            print("Pre-trained weights loaded successfully.")

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.C.parameters(), self.lr, [self.beta1, self.beta2])

        # Print network architecture
        self.print_network(self.C, 'Classifier')

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.C.cuda()
  #=======================================================================================#
  #=======================================================================================#
  def print_network(self, model, name):
      num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(name)
      print("The number of parameters: {}".format(num_params))

  #=======================================================================================#
  #=======================================================================================#
  def load_pretrained_model(self):
    model = self.pretrained_model
    self.C.load_state_dict(torch.load(model))
    print('loaded trained model: {}'.format(model))

  #=======================================================================================#
  #=======================================================================================#
  def build_tensorboard(self):
    from logger import Logger
    self.logger = Logger(self.log_path)

  #=======================================================================================#
  #=======================================================================================#
  def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

  #=======================================================================================#
  #=======================================================================================#
  def reset_grad(self):
    self.optimizer.zero_grad()

  #=======================================================================================#
  #=======================================================================================#
  def to_var(self, x, volatile=False):
    if torch.cuda.is_available():
      x = x.cuda()
    return Variable(x, volatile=volatile)

  #=======================================================================================#
  #=======================================================================================#
  def threshold(self, x):
    x = x.clone()
    x = (x >= 0.5).float()
    return x

  #=======================================================================================#
  #=======================================================================================#
  def denorm(self, x):
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    out = x + mean
    return out.clamp_(0, 1)

  #=======================================================================================#
  #=======================================================================================#
  def blurRANDOM(self, img):
    self.blurrandom +=1

    np.random.seed(self.blurrandom) 
    gray = np.random.randint(0,2,img.size(0))
    np.random.seed(self.blurrandom)
    sigma = np.random.randint(2,9,img.size(0))
    np.random.seed(self.blurrandom)
    window = np.random.randint(7,29,img.size(0))

    trunc = (((window-1)/2.)-0.5)/sigma
    # ipdb.set_trace()
    conv_img = torch.zeros_like(img.clone())
    for i in range(img.size(0)):    
      # ipdb.set_trace()
      if gray[i] and self.GRAY:
        conv_img[i] = torch.from_numpy(filters.gaussian_filter(img[i], sigma=sigma[i], truncate=trunc[i]))
      else:
        for j in range(img.size(1)):
          conv_img[i,j] = torch.from_numpy(filters.gaussian_filter(img[i,j], sigma=sigma[i], truncate=trunc[i]))

    return conv_img
    
  #=======================================================================================#
  #=======================================================================================#
  def plot_cm(self, CM, aca_val, E, i):
    if isinstance(CM, np.ndarray):
        plot_confusion_matrix(CM, classes=self.class_names, normalize=True,
                              title='Normalized confusion matrix (ACC: %0.3f)' % aca_val,
                              save_path=os.path.join(self.model_save_path, 'CM_%s_%s.png'%(E, i)))
    else:
        print("CM is not a numpy array, skipping plotting.")

  #=======================================================================================#
  #=======================================================================================#
  def train(self):
    # Set dataloader

    # The number of iterations per epoch
    iters_per_epoch = len(self.data_loader)
    data_loader = self.data_loader

    # lr cache for decaying
    lr = self.lr
    
    # Start with trained model if exists
    if self.pretrained_model:
        start = int(self.pretrained_model.split('_')[1].split('/')[2])
        # Decay learning rate
        for i in range(start):
            if (i+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                self.update_lr(lr)
                print('Decay learning rate to: {}.'.format(lr))      
    else:
        start = 0

    last_model_step = len(self.data_loader)

    print("Log path: "+self.log_path)

    Log = "[EmoNets] bs:{}, fold:{}, GPU:{}, !{}, from:{}".format(self.batch_size, self.fold, self.GPU, self.mode_data, self.finetuning) 
    loss_cum = {}
    loss_cum['LOSS'] = []
    flag_init=True   

    loss_val_prev = 90
    aca_val_prev = 0
    non_decreasing = 0

    # Start training
    start_time = time.time()

    for e in range(start, self.num_epochs):
        E = str(e+1).zfill(2)
        self.C.train()

        if flag_init:
            ACC_val, aca_val, loss_val = self.val(init=True)  
            log = '[ACA_VAL: %0.3f LOSS_VAL: %0.3f]'%(aca_val, loss_val)
            print(log)
            flag_init = False
            if self.pretrained_model:
                aca_val_prev=aca_val
            #self.plot_cm(CM, aca_val, E, 0) 

        for i, (rgb_img, rgb_label) in tqdm.tqdm(enumerate(self.data_loader), \
                total=len(self.data_loader), desc='Epoch: %d/%d | %s'%(e,self.num_epochs, Log)):
            # ipdb.set_trace()
            if self.BLUR: 
                rgb_img = self.blurRANDOM(rgb_img)

            rgb_img = self.to_var(rgb_img)
            
            # Dynamically handle the dimensionality of rgb_label
            if len(rgb_label.shape) > 1:
                rgb_label = rgb_label.squeeze(1)

            rgb_label = self.to_var(rgb_label)

            out = self.C(rgb_img)

            loss_cls = self.LOSS(out, rgb_label)   

            # Backward + Optimize
            self.reset_grad()
            loss_cls.backward()
            self.optimizer.step()

            # Logging
            loss = {}
            loss['LOSS'] = loss_cls.item()
            loss_cum['LOSS'].append(loss_cls.item())    

            # Print out log info
            if (i+1) % self.log_step == 0 or (i+1) == last_model_step:
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

        #F1 val
        ACC_val, aca_val, loss_val = self.val()

        if self.use_tensorboard:
            # Log validation metrics to tensorboard
            self.logger.scalar_summary('ACC_val: ', aca_val, e * iters_per_epoch + i + 1) 
            self.logger.scalar_summary('LOSS_val: ', loss_val, e * iters_per_epoch + i + 1)     

            for tag, value in loss_cum.items():
                self.logger.scalar_summary(tag, np.array(value).mean(), e * iters_per_epoch + i + 1)   

        # Print validation metrics
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = 'Elapsed: %s | [ACC_VAL: %0.3f LOSS_VAL: %0.3f] | Train'%(elapsed, aca_val, loss_val)
        for tag, value in loss_cum.items():
            log += ", {}: {:.4f}".format(tag, np.array(value).mean())   

        print(log)

        # Save model if validation accuracy improves
        if aca_val > aca_val_prev:        
            torch.save(self.C.state_dict(), os.path.join(self.model_save_path, '{}_{}.pth'.format(E, i+1)))
            print("! Saving model")
            # Compute confusion matrix
            # np.set_printoptions(precision=2)
            # self.plot_cm(CM, aca_val, E, i+1)       
                
        #Stats per epoch
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = 'Elapsed: %s | [ACC_VAL: %0.3f LOSS_VAL: %0.3f] | Train'%(elapsed, aca_val, loss_val)
        for tag, value in loss_cum.items():
            log += ", {}: {:.4f}".format(tag, np.array(value).mean())   

        print(log)

        # Save model if validation accuracy improves
        if aca_val > aca_val_prev:        
            torch.save(self.C.state_dict(), os.path.join(self.model_save_path, '{}_{}.pth'.format(E, i+1)))
            print("! Saving model")
            # Compute confusion matrix
            # np.set_printoptions(precision=2)
           # self.plot_cm(CM, aca_val, E, i+1)       

            aca_val_prev = aca_val
            non_decreasing = 0

        else:
            non_decreasing += 1
            if non_decreasing == self.stop_training:
                print("During {} epochs ACC VAL was not increasing.".format(self.stop_training))
                return

        # Decay learning rate
        if (e+1) > (self.num_epochs - self.num_epochs_decay):
            lr -= (self.lr / float(self.num_epochs_decay))
            self.update_lr(lr)
            print ('Decay learning rate to: {}.'.format(lr))

  #=======================================================================================#
  #=======================================================================================#
  def val(self, init=False, load=False, plot=False):
    # Load trained parameters
    if init:
        from data_loader import get_loader
        self.data_loader_val = get_loader(self.metadata_path, self.image_size,
                                          self.batch_size, 'val', self.fold)
        txt_path = os.path.join(self.model_save_path, '0_init_val.txt')

    if load:
        self.data_loader_val = self.data_loader
        last_file = sorted(glob.glob(os.path.join(self.model_save_path, '*.pth')))[-1]
        last_name = os.path.basename(last_file).split('.')[0]
        txt_path = os.path.join(self.model_save_path, '{}_{}_val.txt'.format(last_name, '{}'))
        try:
            output_txt = sorted(glob.glob(txt_path.format('*')))[-1]
            number_file = len(glob.glob(output_txt))
        except:
            number_file = 0
        txt_path = txt_path.format(str(number_file).zfill(2))

        D_path = os.path.join(self.model_save_path, '{}.pth'.format(last_name))
        self.C.load_state_dict(torch.load(D_path))

    self.C.eval()

    if load:
        self.f = open(txt_path, 'a')
    acc, aca, loss = self.ACC_TEST(self.data_loader_val, verbose=load)
    if load:
        self.f.close()

    CM = np.zeros((self.num_classes, self.num_classes))  # Initialize confusion matrix

    if plot:
        with torch.no_grad():
            for images, labels in self.data_loader_val:
                images = self.to_var(images)
                labels = self.to_var(labels)
                outputs = self.C(images)
                _, predicted = torch.max(outputs, 1)
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    predicted_label = predicted[i].item()
                    CM[true_label][predicted_label] += 1

        np.set_printoptions(precision=2)

    return acc, aca, loss
  
  #=======================================================================================#
  #=======================================================================================#
  def sample(self):
    """Get a dataset sample."""
    import math
    for i, (rgb_img, rgb_label, rgb_files) in enumerate(self.data_loader):
        # ipdb.set_trace()
        if self.BLUR: rgb_img = self.blurRANDOM(rgb_img)
        img_file = 'show/%s.jpg'%(str(i).zfill(4))
        save_image(self.denorm(rgb_img), img_file, nrow=8)
        if i==25: break



