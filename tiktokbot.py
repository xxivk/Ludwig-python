import torch 
print(torch.cuda.is_available())
print(torch.zeros(1).cuda())
torch.backends.cudnn.enabled == True
print(torch.backends.cudnn.version())
import multiprocessing

num_workers = multiprocessing.cpu_count()
print("Number of available CPU cores: ", num_workers)
import torch

num_gpus = torch.cuda.device_count()
print("Number of available GPUs: ", num_gpus)

#a=torch.cuda.FloatTensor()
#print(a)

import torch
#from fastai.vision import *
#from fastai.metrics import error_rate
#
#print("Is cuda available?", torch.cuda.is_available())
#
#print("Is cuDNN version:", torch.backends.cudnn.version())
#
#print("cuDNN enabled? ", torch.backends.cudnn.enabled)

#x = torch.rand(5, 3)
#print(x)
#import torch
#print(torch.backends.cudnn.enabled)
#
#print(torch.cuda.is_available())


#from torch import nn
#net = nn.Sequential(
#    nn.Linear(18*18, 80),
#    nn.ReLU(),
#    nn.Linear(80, 80),
#    nn.ReLU(),
#    nn.Linear(80, 10),
#    nn.LogSoftmax()
#).cuda()
#conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia    


# Install Ray with support for the dashboard + cluster launcher
#pip install -U "ray[default]"
#
#pip install -U "ray[air]" # installs Ray + dependencies for Ray AI Runtime
#pip install -U "ray[tune]"  # installs Ray + dependencies for Ray Tune
#pip install -U "ray[rllib]"  # installs Ray + dependencies for Ray RLlib
#pip install -U "ray[serve]"  # installs Ray + dependencies for Ray Serve

# Install Ray with minimal dependencies
# pip install -U ray
#pip install -U "ray[tune]
#pip install -U "ray[air]