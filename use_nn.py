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

  #model_filename = 'runs_saved/run2_loss201/model_epoch_2600.pt' # disco
  #eval_filename = 'nn_loss201_run2.root'
  #nvar = 10

  #model_filename = 'runs_saved/run2_loss203/model_epoch_2300.pt' # disco + signi
  #eval_filename = 'nn_loss203_run2.root'
  #nvar = 10

  #model_filename = 'runs/May20_01-59-34_hepmacprojb.local/model_epoch_4990.pt'
  #model_filename = 'runs/May21_13-12-12_cms37/model_epoch_24900.pt'
  #model_filename = 'runs/May22_19-48-24_cms37/model_epoch_1700.pt'
  #model_filename = 'runs/May26_12-09-01_cms37/model_epoch_1100.pt' # disco 10
  #model_filename = 'runs/May26_17-33-58_cms37/model_epoch_1000.pt' # disco 20
  model_filename = 'runs/May27_06-48-55_cms37/model_epoch_200.pt' # disco 40
  eval_filename = 'nn_loss203_i11_fullwin_run2.root'
  nvar = 11


  state_dict = torch.load(model_filename)
  #model = SimpleNetwork(input_size=10, hidden_size=40, output_size=1).to(device)
  #model = SimpleNetwork(input_size=11, hidden_size=44, output_size=1).to(device)
  model = SimpleNetwork(input_size=nvar, hidden_size=nvar*4, output_size=1).to(device)
  model.load_state_dict(state_dict)
  print('Loaded '+model_filename)

  if nvar == 11:
    feature_names = ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 
                     #'photon_res', 
                     'llg_mass_err',
                     'photon_rapidity', 'l1_rapidity', 'l2_rapidity',
                     'llg_flavor', 'gamma_pt']
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
                        [15.015657, 295.22623], #gamma_pt
                        ]
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

  #loss_fn = z_loss()

  #train_dataset = RootDataset(root_filename='ntuples_mva/TMVA_nn.root',
  #                          tree_name = "dataset/TrainTree",
  #                          features = ['photon_mva', 
  #                          'min_dR', 
  #                          'pt_mass', 
  #                          'cosTheta',
  #                          'costheta', 
  #                          'photon_res', 
  #                          'photon_rapidity', 
  #                          'l1_rapidity', 
  #                          'l2_rapidity'],
  #                          normalize = normalize_max_min,
  #                          cut = '1',
  #                          spectators = ['llg_mass', 'w_lumi'],
  #                          class_branch = ['classID'],
  #                          entry_stop = 12960)
  train_dataset = RootDataset(root_filename='train_sample_run2_winfull.root',
                            tree_name = "train_tree",
                            features = feature_names,
                            normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumi'],
                            class_branch = ['classID'])
  print(f'train entries: {len(train_dataset)}')

  #test_dataset = RootDataset(root_filename='ntuples_mva/TMVA_nn.root',
  #                          tree_name = "dataset/TestTree",
  #                          features = ['photon_mva', 
  #                          'min_dR', 
  #                          'pt_mass', 
  #                          'cosTheta',
  #                          'costheta', 
  #                          'photon_res', 
  #                          'photon_rapidity', 
  #                          'l1_rapidity', 
  #                          'l2_rapidity'],
  #                          normalize = normalize_max_min,
  #                          cut = '1',
  #                          spectators = ['llg_mass', 'w_lumi'],
  #                          class_branch = ['classID'], 
  #                          entry_stop = len(train_dataset))
  test_dataset = RootDataset(root_filename='test_sample_run2_winfull.root',
                            tree_name = "test_tree",
                            features = feature_names,
                            normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumi'],
                            class_branch = ['classID'])
  print(f'test entries: {len(test_dataset)}')

  eval_dataset = RootDataset(root_filename='test_full_sample_run2_winfull.root',
                            tree_name = "test_tree",
                            features = feature_names,
                            normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumi'],
                            class_branch = ['classID'])
  print(f'eval entries: {len(eval_dataset)}')

  #batch_size = 128
  #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
  #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  #filename = 'trash/evaluate.root'
  #results = {'train': {}, 'test': {}}
  #results['train'] = evaluate_sample(train_dataloader, device, model, loss_fn, True)
  #results['test'] = evaluate_sample(test_dataloader, device, model, loss_fn, True)
  ## Create evaluation root file
  #with uproot.recreate(filename) as root_file:
  #  root_file['train_tree'] = {'x': results['train']['x'], 'y': results['train']['y'], 'yhat': results['train']['yhat'], 'mass': results['train']['mass'], 'weight': results['train']['weight']}
  #  root_file['test_tree'] = {'x': results['test']['x'], 'y': results['test']['y'], 'yhat': results['test']['yhat'], 'mass': results['test']['mass'], 'weight': results['test']['weight']}
  #mva_dict = [evaluate.load_mva_dict(filename, 'mva')]
  ## Evaluate train
  ##train_significances, train_purities = evaluate.evaluate_significance_with_resolution(mva_dict, draw=False, tree_type='train_tree')
  #train_significances, train_purities = evaluate.evaluate_significance(mva_dict, draw=False, tree_type='train_tree')
  #train_significances_with_res, train_purities_with_res = evaluate.evaluate_significance_with_resolution(mva_dict, draw=False, tree_type='train_tree')
  #train_std_divs = evaluate.evaluate_correlation(mva_dict, draw=False, tree_type='train_tree')
  #print('train sig', train_significances)
  #print('train pur', train_purities)
  #print('train sig res', train_significances_with_res)
  #print('train pur res', train_purities_with_res)
  #print('train std dev', train_std_divs)

  #eval_dataset = RootDataset(root_filename='ntuples_mva/TMVA_nn.root',
  #                          tree_name = "dataset/TestTree",
  #                          features = ['photon_mva', 
  #                          'min_dR', 
  #                          'pt_mass', 
  #                          'cosTheta',
  #                          'costheta', 
  #                          'photon_res', 
  #                          'photon_rapidity', 
  #                          'l1_rapidity', 
  #                          'l2_rapidity'],
  #                          normalize = normalize_max_min,
  #                          cut = '1',
  #                          spectators = ['llg_mass', 'w_lumi'],
  #                          class_branch = ['classID'])
  #print(f'eval entries: {len(test_dataset)}')

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
