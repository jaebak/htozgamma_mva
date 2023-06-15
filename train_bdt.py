#!/usr/bin/env python3
import ROOT
import torch.utils.data
import sklearn.metrics
import sklearn.ensemble
import xgboost
from RootDataset import RootDataset
import uproot
import numpy as np

if __name__ == '__main__':

  # weight_types: 1 is decorr, 2 is use weight, 3 is weight equally, 4 is only shape decorr, 5 is no weight
  weight_type = 3

  feature_names = ['min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi',
                   'photon_rapidity', 'l1_rapidity', 'l2_rapidity',
                   'llg_flavor', 'llg_ptt'
                   ]

  use_full_mass_windows = False
  use_weight = True
  if weight_type == 1:
    use_full_mass_windows = True
    gbdt_filename = 'mva_output_ntuples/gbdt_run2_decorr.root'
    xgbt_filename = 'mva_output_ntuples/xgbdt_run2_decorr.root'
  elif weight_type == 2:
    gbdt_filename = 'mva_output_ntuples/gbdt_run2.root'
    xgbt_filename = 'mva_output_ntuples/xgbdt_run2.root'
  elif weight_type == 3:
    gbdt_filename = 'mva_output_ntuples/gbdt_wequal_run2.root'
    xgbt_filename = 'mva_output_ntuples/xgbdt_wequal_run2.root'
  elif weight_type == 4:
    use_full_mass_windows = True
    gbdt_filename = 'mva_output_ntuples/gbdt_run2_decorr_shape.root'
    xgbt_filename = 'mva_output_ntuples/xgbdt_run2_decorr_shape.root'
  elif weight_type == 5:
    gbdt_filename = 'mva_output_ntuples/gbdt_run2_noweight.root'
    xgbt_filename = 'mva_output_ntuples/xgbdt_run2_noweight.root'
    use_weight = False

  if use_full_mass_windows:
    train_filename = 'mva_input_ntuples/train_sample_run2_lumi_winfull.root'
    test_filename = 'mva_input_ntuples/test_sample_run2_lumi_winfull.root'
    test_full_filename = 'mva_input_ntuples/test_full_sample_run2_lumi_winfull.root'
  else:
    train_filename = 'mva_input_ntuples/train_sample_run2_lumi.root'
    test_filename = 'mva_input_ntuples/test_sample_run2_lumi.root'
    test_full_filename = 'mva_input_ntuples/test_full_sample_run2_lumi.root'

  train_dataset = RootDataset(root_filename= train_filename,
                            tree_name = "train_tree",
                            features = feature_names,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumiXyearXshape', 'w_lumiXyear', 'classID', 'w_llg_mass'],
                            class_branch = ['classID'])
  print(f'train entries: {len(train_dataset)}')

  train_feature_array = train_dataset.feature_array
  train_hot_label_array = train_dataset.label_array
  train_label_array = train_dataset.label_array[:,1]
  train_spec_array = train_dataset.spec_array
  train_mass_array = train_dataset.spec_array[:,0]
  if weight_type == 1: 
    train_weight_array = train_dataset.spec_array[:,1]
  elif weight_type == 3:
    nbkg = np.count_nonzero(train_dataset.spec_array[:,3]==0)
    nsig = np.count_nonzero(train_dataset.spec_array[:,3]==1)
    nsig_nbkg = nsig * 1./ nbkg
    # weight signal and bkg equally
    train_weight_array = np.array(train_dataset.spec_array[:,3])
    train_weight_array[train_weight_array == 0] = nsig_nbkg
    print(f'nsig: {nsig} nbkg: {nbkg}. Reweight bkg by {nsig_nbkg} sum: {np.sum(train_weight_array)}')
  elif weight_type == 2: 
    # weight according to weight
    train_weight_array = np.array(train_dataset.spec_array[:,2])
  elif weight_type == 4: 
    # weight according to weight
    train_weight_array = np.array(train_dataset.spec_array[:,4])
  train_w_lumi_array = train_dataset.spec_array[:,2]
  nlabels = train_hot_label_array.shape[1]
  if use_weight:
    # Set negative weights to 0
    train_weight_array[train_weight_array<0] = 0.

  # Train xgboost
  print("Training xgboost")
  xgbdt_classifier = xgboost.XGBClassifier(max_depth=3, n_estimators=100)
  if use_weight: xgbdt_classifier.fit(train_feature_array, train_hot_label_array[:,nlabels-1], sample_weight=train_weight_array)
  else: xgbdt_classifier.fit(train_feature_array, train_hot_label_array[:,nlabels-1])

  # Evaluate
  test_dataset = RootDataset(root_filename=test_filename,
                            tree_name = "test_tree",
                            features = feature_names,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumiXyear'],
                            class_branch = ['classID'])
  print(f'test entries: {len(test_dataset)}')
  test_feature_array = test_dataset.feature_array
  test_hot_label_array = test_dataset.label_array
  test_label_array = test_dataset.label_array[:,1]
  test_spec_array = test_dataset.spec_array
  test_mass_array = test_dataset.spec_array[:,0]
  test_weight_array = test_dataset.spec_array[:,1]
  nlabels = test_label_array.shape

  eval_dataset = RootDataset(root_filename=test_full_filename,
                            tree_name = "test_tree",
                            features = feature_names,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumiXyear'],
                            class_branch = ['classID'])
  print(f'eval entries: {len(eval_dataset)}')
  eval_feature_array = eval_dataset.feature_array
  eval_hot_label_array = eval_dataset.label_array
  eval_label_array = eval_dataset.label_array[:,1]
  eval_spec_array = eval_dataset.spec_array
  eval_mass_array = eval_dataset.spec_array[:,0]
  eval_weight_array = eval_dataset.spec_array[:,1]


  train_weight_array = train_dataset.spec_array[:,2]

  # xgboost
  # Predict
  test_predict_array_xgbdt_raw = xgbdt_classifier.predict_proba(test_feature_array)
  test_predict_array_xgbdt = test_predict_array_xgbdt_raw[:,1]
  train_predict_array_xgbdt_raw = xgbdt_classifier.predict_proba(train_feature_array)
  train_predict_array_xgbdt = train_predict_array_xgbdt_raw[:,1]
  eval_predict_array_xgbdt_raw = xgbdt_classifier.predict_proba(eval_feature_array)
  eval_predict_array_xgbdt = eval_predict_array_xgbdt_raw[:,1]
  # Save tree
  xgbt_root_file = uproot.recreate(xgbt_filename)
  xgbt_root_file["test_tree"] = {'x': test_feature_array, 'y': test_label_array, 'yhat': test_predict_array_xgbdt, 'mass': test_mass_array, 'weight': test_weight_array}
  xgbt_root_file["train_tree"] = {'x': train_feature_array, 'y': train_label_array, 'yhat': train_predict_array_xgbdt, 'mass': train_mass_array, 'weight': train_weight_array}
  xgbt_root_file["test_full_tree"] = {'x': eval_feature_array, 'y': eval_label_array, 'yhat': eval_predict_array_xgbdt, 'mass': eval_mass_array, 'weight': eval_weight_array}
  print('Wrote xgbdt results to '+xgbt_filename)

