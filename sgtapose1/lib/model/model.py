from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.dla import DLA_PlanAAblation, DLASeg, DLA_PlanA, DLA_PlanAWindow, DLA_PlanACAT, DLA_PlanAWindow_Three
from .networks.dla import DLA_PlanAWindow_l3new
from .networks.hourglass import DreamHourglass, ResnetSimple

_network_factory = {
  "dreamhourglass" : DreamHourglass,
  "dlapa" : DLA_PlanA,
  "dlapawd" : DLA_PlanAWindow,
  "dlapawd3" : DLA_PlanAWindow_Three,
  "dlapawdl3new" : DLA_PlanAWindow_l3new, 
  "dlapacat" : DLA_PlanACAT,
  "dlaabla" : DLA_PlanAAblation
}

def create_model(arch, head, head_conv, opt=None):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  model_class = _network_factory[arch]
  model = model_class(num_layers, heads=head, head_convs=head_conv, opt=opt)
  return model

def create_dream_hourglass(n_keypoints=7,mode="vgg", deconv_decoder=False):
    n_image_input_channels = 3
    if mode == "vgg":  
        # model = torch.nn.DataParallel(DreamHourglass(n_keypoints=n_keypoints, n_image_input_channels=n_image_input_channels,\
    # deconv_decoder=deconv_decoder))
        model = DreamHourglass(n_keypoints=n_keypoints, n_image_input_channels=n_image_input_channels,\
    deconv_decoder=deconv_decoder)
    elif mode == "resnet":
        # model = torch.nn.DataParallel(ResnetSimple(n_keypoints=n_keypoints, full=deconv_decoder))
        model = ResnetSimple(n_keypoints=n_keypoints, full=deconv_decoder)
    return model

def load_model(model, model_path, opt, optimizer=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
   
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  for k in state_dict:
    # print("k",k)
    if k in model_state_dict:
      if (state_dict[k].shape != model_state_dict[k].shape) or \
        (opt.reset_hm and k.startswith('hm') and (state_dict[k].shape[0] in [80, 1])):
        if opt.reuse_hm:
          print('Reusing parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          if state_dict[k].shape[0] < state_dict[k].shape[0]:
            model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
          else:
            model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
          state_dict[k] = model_state_dict[k]
        else:
          print('Skip loading parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k))
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and opt.resume:
    if 'optimizer' in checkpoint:
      # optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = opt.lr
      for step in opt.lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

