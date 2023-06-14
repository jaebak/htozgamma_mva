#!/usr/bin/env python3
import ROOT
from torch import nn
import torch.utils.data
import torch
import uproot
import numpy as np
import xml.etree.ElementTree as ET
from array import array
import math
import copy
import sklearn.metrics
import sklearn.ensemble
import matplotlib.pyplot as plt
import xgboost
from torch.utils.tensorboard import SummaryWriter
import evaluate
import train_nn
from train_nn import SimpleNetwork
from train_nn import z_loss
from train_nn import RootDataset
from train_nn import evaluate_sample
from train_nn import unnormalize

if __name__ == "__main__":
  device = "cpu"
  torch.manual_seed(1)

  #model_filename = 'runs/Jun12_02-44-05_cms37/model_epoch_2400.pt' # loss6
  #eval_filename = 'mva_output_ntuples/nn_loss6_run2.root'
  #nvar = 11

  model_filename = 'runs/Jun12_02-44-52_cms37/model_epoch_10000.pt' # loss0
  eval_filename = 'mva_output_ntuples/nn_loss0_run2.root'
  nvar = 11


  state_dict = torch.load(model_filename)
  model = SimpleNetwork(input_size=nvar, hidden_size=nvar*4, output_size=1).to(device)
  model.load_state_dict(state_dict)
  print('Loaded '+model_filename)

  if nvar == 12:
    feature_names = ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 
                     #'photon_res', 
                     'llg_mass_err',
                     'photon_rapidity', 'l1_rapidity', 'l2_rapidity',
                     'llg_flavor', 'max_dR', 'llg_ptt']
    normalize_max_min = [[-0.57861328125,0.98583984375],
                        [0.400207489729,3.32512640953],
                        [0.000612989999354,4.14180803299],
                        [-0.999573588371,0.998835206032],
                        [-0.987939178944,0.983025610447],
                        #[0.00963300466537,1.51448833942], # photon_res
                        [0.53313,16.254], # llg_mass_err
                        [-2.49267578125,2.4921875],
                        [-2.49072265625,2.4814453125],
                        [-2.49072265625,2.50830078125],
                        [1., 2.],
                        [0.41682249, 5.1650324],
                        [0., 396.69781],
                        ]
  elif nvar == 11:
    feature_names = ['min_dR', 'pt_mass', 'pt_mass', 'cosTheta', 'costheta', 'phi',
                     'photon_rapidity', 'l1_rapidity', 'l2_rapidity',
                     'llg_flavor', 'llg_ptt']
    normalize_max_min = [[0.40017,3.3237],
                         [0.49768,4.1958],
                         [0.13637,1.8901],
                        [-0.99999,0.99992],
                        [-0.98130,0.97890],
                        [1.4860e-05,6.2832],
                        [-2.4998,2.4999],
                        [-2.5000,2.4999],
                        [-2.4998,2.4998],
                        [0., 1.],
                        [0.00016470, 291.34]]
  elif nvar == 10:
    feature_names = ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 
                     #'photon_res', 
                     'llg_mass_err',
                     'photon_rapidity', 'l1_rapidity', 'l2_rapidity',
                     'llg_flavor']
    normalize_max_min = [[-0.57861328125,0.98583984375],
                        [0.400207489729,3.32512640953],
                        [0.000612989999354,4.14180803299],
                        [-0.999573588371,0.998835206032],
                        [-0.987939178944,0.983025610447],
                        #[0.00963300466537,1.51448833942],
                        [0.53313,16.254], # llg_mass_err
                        [-2.49267578125,2.4921875],
                        [-2.49072265625,2.4814453125],
                        [-2.49072265625,2.50830078125],
                        [1., 2.],
                        ]
  elif nvar == 9:
    feature_names = ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 
                     'photon_res', 
                     'photon_rapidity', 'l1_rapidity', 'l2_rapidity',]
    normalize_max_min = [[-0.57861328125,0.98583984375],
                        [0.400207489729,3.32512640953],
                        [0.000612989999354,4.14180803299],
                        [-0.999573588371,0.998835206032],
                        [-0.987939178944,0.983025610447],
                        [0.00963300466537,1.51448833942],
                        [-2.49267578125,2.4921875],
                        [-2.49072265625,2.4814453125],
                        [-2.49072265625,2.50830078125],
                        ]

  train_dataset = RootDataset(root_filename='mva_input_ntuples/train_sample_run2_lumi.root',
                            tree_name = "train_tree",
                            features = feature_names,
                            normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumiXyear'],
                            class_branch = ['classID'])
  print(f'train entries: {len(train_dataset)}')

  test_dataset = RootDataset(root_filename='mva_input_ntuples/test_sample_run2_lumi.root',
                            tree_name = "test_tree",
                            features = feature_names,
                            normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumiXyear'],
                            class_branch = ['classID'])
  print(f'test entries: {len(test_dataset)}')

  eval_dataset = RootDataset(root_filename='mva_input_ntuples/test_full_sample_run2_lumi.root',
                            tree_name = "test_tree",
                            features = feature_names,
                            normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumiXyear'],
                            class_branch = ['classID'])
  print(f'eval entries: {len(eval_dataset)}')


  train_feature_array = train_dataset.feature_array
  train_hot_label_array = train_dataset.label_array
  train_label_array = train_dataset.label_array[:,1]
  train_spec_array = train_dataset.spec_array
  train_unnorm_feature_array = unnormalize(train_feature_array, normalize_max_min)
  train_mass_array = train_dataset.spec_array[:,0]
  train_weight_array = train_dataset.spec_array[:,1]
  nlabels = train_hot_label_array.shape[1]

  model.eval()
  with torch.no_grad():
    train_predict_array_nn_raw = model(torch.from_numpy(train_feature_array).to(device)).to('cpu')
    train_predict_array_nn = train_predict_array_nn_raw.squeeze()
  print(f'nn label: {train_hot_label_array[:,nlabels-1]} predict: {train_predict_array_nn}')

  test_feature_array = test_dataset.feature_array
  test_hot_label_array = test_dataset.label_array
  test_label_array = test_dataset.label_array[:,1]
  test_spec_array = test_dataset.spec_array
  test_unnorm_feature_array = unnormalize(test_feature_array, normalize_max_min)
  test_mass_array = test_dataset.spec_array[:,0]
  test_weight_array = test_dataset.spec_array[:,1]

  model.eval()
  with torch.no_grad():
    test_predict_array_nn_raw = model(torch.from_numpy(test_feature_array).to(device)).to('cpu')
    test_predict_array_nn = test_predict_array_nn_raw.squeeze()
  print(f'nn label: {test_hot_label_array[:,nlabels-1]} predict: {test_predict_array_nn}')


  eval_feature_array = eval_dataset.feature_array
  eval_hot_label_array = eval_dataset.label_array
  eval_label_array = eval_dataset.label_array[:,1]
  eval_spec_array = eval_dataset.spec_array
  eval_unnorm_feature_array = unnormalize(eval_feature_array, normalize_max_min)
  eval_mass_array = eval_dataset.spec_array[:,0]
  eval_weight_array = eval_dataset.spec_array[:,1]

  model.eval()
  with torch.no_grad():
    eval_predict_array_nn_raw = model(torch.from_numpy(eval_feature_array).to(device)).to('cpu')
    eval_predict_array_nn = eval_predict_array_nn_raw.squeeze()
  print(f'nn label: {eval_hot_label_array[:,nlabels-1]} predict: {eval_predict_array_nn}')

  root_file = uproot.recreate(eval_filename)
  root_file["test_tree"] = {'x_norm': test_feature_array, 'x': test_unnorm_feature_array, 'y': test_label_array, 'yhat': test_predict_array_nn, 'mass': test_mass_array, 'weight': test_weight_array}
  root_file["train_tree"] = {'x_norm': train_feature_array, 'x': train_unnorm_feature_array, 'y': train_label_array, 'yhat': train_predict_array_nn, 'mass': train_mass_array, 'weight': train_weight_array}
  root_file["test_full_tree"] = {'x_norm': eval_feature_array, 'x': eval_unnorm_feature_array, 'y': eval_label_array, 'yhat': eval_predict_array_nn, 'mass': eval_mass_array, 'weight': eval_weight_array}
  print('Results saved to '+eval_filename)
