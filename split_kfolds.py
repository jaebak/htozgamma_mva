#!/usr/bin/env python3
import uproot
import numpy as np
import os

def split_ntuple(in_filename, in_treename, random_seed, cut, train_fraction, weight_branch_name, out_filename):
  # Set random seed
  np.random.seed(random_seed)

  # Read the ntuples
  in_file = uproot.open(in_filename)
  in_tree = in_file[in_treename]
  branches = in_tree.keys()
  in_ntuples = in_tree.arrays(branches, '1', library='np')
  ntuples = np.stack([np.squeeze(in_ntuples[feat]) for feat in branches], axis=1)

  # Randomize ntuple
  random_idx = np.random.permutation(len(ntuples))
  ntuples = ntuples[random_idx]

  # Split the ntuples
  n_entries = len(ntuples)
  train_n_entries = int(train_fraction * n_entries)
  train_ntuples = ntuples[:train_n_entries]
  test_ntuples = ntuples[train_n_entries:]
  print(f'Total entries: {n_entries} train entries: {len(train_ntuples)}, test entries: {len(test_ntuples)}')

  # Modify the ntuples. Scale up weights
  weight_branch_index = branches.index(in_weightname)
  train_ntuples[:,weight_branch_index] = train_ntuples[:,weight_branch_index] / train_fraction
  test_ntuples[:,weight_branch_index] = test_ntuples[:,weight_branch_index] / (1-train_fraction)

  # Create the ntuples
  out_train_ntuples = {}
  out_test_ntuples = {}
  out_eval_ntuples = {}
  for ibranch, branch in enumerate(branches):
    if branch == 'classID':
      out_test_ntuples[branch] = np.int32(test_ntuples[:,ibranch])
      out_train_ntuples[branch] = np.int32(train_ntuples[:,ibranch])
    else:
      out_test_ntuples[branch] = np.float32(test_ntuples[:,ibranch])
      out_train_ntuples[branch] = np.float32(train_ntuples[:,ibranch])

  #print(f'train: {out_train_ntuples}')
  #print(f'test: {out_test_ntuples}')

  # Create tmp file
  tmp_filename = f'{out_filename}.tmp'
  with uproot.recreate(tmp_filename) as out_file:
    out_file["test_full_tree"] = out_test_ntuples
    out_file["train_full_tree"] = out_train_ntuples
  print('temp tree saved to '+tmp_filename)
  in_file.close()

  # Apply cut to sample using tmp file
  in_file = uproot.open(tmp_filename)
  test_full_tree = in_file['test_full_tree']
  test_full_ntuples = test_full_tree.arrays(branches, library='np')
  if cut == '1': test_ntuples = test_full_tree.arrays(branches, library='np')
  else: test_ntuples = test_full_tree.arrays(branches, cut, library='np')
  train_full_tree = in_file['train_full_tree']
  train_full_ntuples = train_full_tree.arrays(branches, library='np')
  if cut == '1': train_ntuples = train_full_tree.arrays(branches, library='np')
  else: train_ntuples = train_full_tree.arrays(branches, cut, library='np')
  #print(f'train_full: {train_full_ntuples}')
  #print(f'train: {train_ntuples}')
  with uproot.recreate(out_filename) as out_file:
    out_file["train_tree"] = train_ntuples
    out_file["train_full_tree"] = train_full_ntuples
    out_file["test_tree"] = test_ntuples
    out_file["test_full_tree"] = test_full_ntuples
  print('tree saved to '+out_filename)
  in_file.close()

if __name__ == '__main__':

  in_filename = 'train_sample_run2_lumi_winfull.root'
  #in_filename = 'train_sample_run2_lumi.root'
  in_treename = 'train_tree'
  in_weightname = 'w_lumiXyear'
  train_fraction = 0.8
  num_kfolds = 10
  train_cut = '(llg_mass>120) & (llg_mass<130)'
  output_base_filename = 'train_sample_run2_lumi'
  #train_cut = '1'
  #output_filename = 'train_sample_run2_lumi_winfull'

  for ifold in range(num_kfolds):
    random_seed = ifold+1
    output_filename = f'ntuples_kfold/{output_base_filename}_s{random_seed}.root'
    split_ntuple(in_filename, in_treename, random_seed, train_cut, train_fraction, in_weightname, output_filename)
