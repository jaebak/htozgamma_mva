#!/usr/bin/env python3
import ROOT
import torch.utils.data
import sklearn.metrics
import sklearn.ensemble
import xgboost
from RootDataset import RootDataset
import uproot

if __name__ == '__main__':
  feature_names = ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 
                   'photon_res', 
                   #'llg_mass_err',
                   'photon_rapidity', 'l1_rapidity', 'l2_rapidity',
                   #'llg_flavor',
                   ]
  train_filename = 'train_sample_run2.root'
  test_filename = 'test_sample_run2.root'
  test_full_filename = 'test_full_sample_run2.root'
  gbdt_filename = 'ntuples_mva/gbdt_run2.root'
  xgbt_filename = 'ntuples_mva/xgbdt_run2.root'

  train_dataset = RootDataset(root_filename= train_filename,
                            tree_name = "train_tree",
                            features = feature_names,
                            #normalize = normalize_max_min,
                            cut = '1',
                            #spectators = ['llg_mass', 'w_lumi'],
                            spectators = ['llg_mass', 'w_llg_mass', 'w_lumi'],
                            class_branch = ['classID'])
  print(f'train entries: {len(train_dataset)}')

  train_feature_array = train_dataset.feature_array
  train_hot_label_array = train_dataset.label_array
  train_label_array = train_dataset.label_array[:,1]
  train_spec_array = train_dataset.spec_array
  train_mass_array = train_dataset.spec_array[:,0]
  train_weight_array = train_dataset.spec_array[:,1]
  train_w_lumi_array = train_dataset.spec_array[:,2]
  nlabels = train_hot_label_array.shape[1]
  # Set negative weights to 0
  train_weight_array[train_weight_array<0] = 0.

  # Train gradient boosted decision tree
  print("Training gradient BDT")
  gbdt_classifier = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
  gbdt_classifier.fit(train_feature_array, train_hot_label_array[:,nlabels-1], train_weight_array)

  # Train xgboost
  print("Training xgboost")
  xgbdt_classifier = xgboost.XGBClassifier(max_depth=3, n_estimators=100)
  xgbdt_classifier.fit(train_feature_array, train_hot_label_array[:,nlabels-1], sample_weight=train_weight_array)

  # Evaluate
  test_dataset = RootDataset(root_filename=test_filename,
                            tree_name = "test_tree",
                            features = feature_names,
                            #normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumi'],
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
                            spectators = ['llg_mass', 'w_lumi'],
                            class_branch = ['classID'])
  print(f'eval entries: {len(eval_dataset)}')
  eval_feature_array = eval_dataset.feature_array
  eval_hot_label_array = eval_dataset.label_array
  eval_label_array = eval_dataset.label_array[:,1]
  eval_spec_array = eval_dataset.spec_array
  eval_mass_array = eval_dataset.spec_array[:,0]
  eval_weight_array = eval_dataset.spec_array[:,1]



  # Boosted decision tree
  # Predict
  test_predict_array_gbdt_raw = gbdt_classifier.predict_proba(test_feature_array)
  test_predict_array_gbdt = test_predict_array_gbdt_raw[:,1]
  train_predict_array_gbdt_raw = gbdt_classifier.predict_proba(train_feature_array)
  train_predict_array_gbdt = train_predict_array_gbdt_raw[:,1]
  eval_predict_array_gbdt_raw = gbdt_classifier.predict_proba(eval_feature_array)
  eval_predict_array_gbdt = eval_predict_array_gbdt_raw[:,1]
  # Save tree
  gbdt_root_file = uproot.recreate(gbdt_filename)
  gbdt_root_file["test_tree"] = {'x': test_feature_array, 'y': test_label_array, 'yhat': test_predict_array_gbdt, 'mass': test_mass_array, 'weight': test_weight_array}
  gbdt_root_file["train_tree"] = {'x': train_feature_array, 'y': train_label_array, 'yhat': train_predict_array_gbdt, 'mass': train_mass_array, 'weight': train_w_lumi_array}
  gbdt_root_file["test_full_tree"] = {'x': eval_feature_array, 'y': eval_label_array, 'yhat': eval_predict_array_gbdt, 'mass': eval_mass_array, 'weight': eval_weight_array}
  print('Wrote bdt results to '+gbdt_filename)

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

